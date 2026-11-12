"""Strict hot-cache reader for split-screen MiniMax-H3 Ref2VA LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from cocode_world.io import sha256_file
from cocode_world.model import build_ref_video_layout
from cocode_world.reference import EXTERNAL_REFERENCE, TARGET_REFERENCE, validate_external_reference
from cocode_world.proxy import validate_proxy_profile, proxy_latent_timestamps


REF_VIDEO_CACHE_SCHEMA_VERSION = "h3-ref-image-video-splitscreen-cache-v2"
REF_VIDEO_MAIN_TENSORS = {
    "clean_target_video_latent",
    "clean_reference_image_latent",
    "clean_reference_video_latent",
}
REF_VIDEO_TEXT_TENSORS = {"encoder_hidden_states", "text_token_tags"}
REF_VIDEO_REQUIRED_METADATA = {
    "schema_version",
    "sample_id",
    "tensor_path",
    "sha256",
    "text_tensor_path",
    "text_sha256",
    "prompt",
    "original_prompt",
    "scene_description",
    "cp_prompt_suffix",
    "target_video_paths",
    "target_video_sha256",
    "proxy_video_paths",
    "proxy_video_sha256",
    "reference_kinds",
    "reference_has_audio",
    "reference_image_source",
    "reference_image_frozen",
    "split_screen",
    "num_frames",
    "fps",
    "height",
    "width",
    "vae_config_hash",
    "text_encoder_hash",
    "text_encoder_layer",
    "text_encoder_execution_revision",
    "transformer_ref_config_hash",
    "num_text_rows",
    "num_condition_image_rows",
    "num_condition_proxy_video_rows",
    "num_condition_video_rows",
    "num_target_video_rows",
    "num_packed_rows",
    "cache_context_parallel_degree",
    "target_vae_encode_seed",
    "reference_image_vae_encode_seed",
    "reference_video_vae_encode_seed",
}
TARGET_LATENT_SHAPE = (24, 37, 30, 104)
IMAGE_REFERENCE_LATENT_SHAPE = (24, 1, 30, 104)
TARGET_VIDEO_ROWS = 28_860
IMAGE_REFERENCE_ROWS = 780
PROXY_VIDEO_ROWS = 28_860


def cache_geometry(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate the supported canvases and derive their actual latent geometry."""
    height, width = int(metadata["height"]), int(metadata["width"])
    if (int(metadata["num_frames"]), float(metadata["fps"])) != (124, 24.0):
        raise ValueError("Ref2VA cache requires 124 frames at 24 fps")
    if (height, width) not in ((480, 1664), (736, 2560)):
        raise ValueError(f"Unsupported Ref2VA canvas {height}x{width}")
    if (height, width) == (736, 2560):
        if metadata.get("source_canvas") != [720, 2560] or metadata.get("canvas_padding") != {
            "top": 8, "bottom": 8, "left": 0, "right": 0, "mode": "edge"
        }:
            raise ValueError("720p cache must record lossless 720-to-736 edge padding")
    latent_h, latent_w = height // 16, width // 16
    image_rows = (latent_h // 2) * (latent_w // 2)
    proxy_profile = validate_proxy_profile(metadata)
    if proxy_profile is not None and (height, width) != (736, 2560):
        raise ValueError("Compact proxy references require the 720p target profile")
    proxy_shape = tuple(proxy_profile["latent_shape"]) if proxy_profile else (24, 37, latent_h, latent_w)
    proxy_rows = proxy_shape[1] * (proxy_shape[2] // 2) * (proxy_shape[3] // 2)
    return {
        "target_latent_shape": (24, 37, latent_h, latent_w),
        "image_latent_shape": (24, 1, latent_h, latent_w),
        "image_rows": image_rows,
        "video_rows": 37 * image_rows,
        "proxy_latent_shape": proxy_shape,
        "proxy_rows": proxy_rows,
        "seam_column": width // 2,
    }


@dataclass(frozen=True)
class CachedH3RefVideoSample:
    """One target with a frozen first-frame image and a color proxy video."""

    clean_target_video_latent: torch.Tensor
    clean_reference_image_latent: torch.Tensor
    clean_reference_video_latent: torch.Tensor
    encoder_hidden_states: torch.Tensor
    text_token_tags: torch.Tensor
    metadata: dict[str, Any]

    def build_layout(self, *, context_parallel_degree: int):
        clock = proxy_latent_timestamps(self.metadata)
        return build_ref_video_layout(
            text_token_tags=self.text_token_tags,
            num_image_reference_latent_frames=self.clean_reference_image_latent.shape[1],
            image_reference_latent_height=self.clean_reference_image_latent.shape[2],
            image_reference_latent_width=self.clean_reference_image_latent.shape[3],
            num_reference_latent_frames=self.clean_reference_video_latent.shape[1],
            reference_latent_height=self.clean_reference_video_latent.shape[2],
            reference_latent_width=self.clean_reference_video_latent.shape[3],
            num_target_latent_frames=self.clean_target_video_latent.shape[1],
            target_latent_height=self.clean_target_video_latent.shape[2],
            target_latent_width=self.clean_target_video_latent.shape[3],
            patch_size=(1, 2, 2),
            video_channels=self.clean_target_video_latent.shape[0],
            audio_channels=32,
            context_parallel_degree=context_parallel_degree,
            reference_latent_timestamps=clock[0] if clock else None,
            reference_duration_seconds=clock[1] if clock else None,
        )


class CachedH3RefVideoDataset:
    """Load the image+proxy split-screen cache without media decode in training."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        expected_vae_hash: str | None = None,
        expected_text_encoder_hash: str | None = None,
        expected_transformer_ref_hash: str | None = None,
        verify_sha256: bool = False,
        trust_validated_paths: bool = False,
        allowed_root: str | Path | None = None,
        allow_external_reference: bool = False,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        self.allowed_root = Path(allowed_root).resolve() if allowed_root else self.manifest.parent.resolve()
        if not self.allowed_root.is_dir():
            raise FileNotFoundError(self.allowed_root)
        self.expected_vae_hash = expected_vae_hash
        self.expected_text_encoder_hash = expected_text_encoder_hash
        self.expected_transformer_ref_hash = expected_transformer_ref_hash
        self.verify_sha256 = bool(verify_sha256)
        self.trust_validated_paths = bool(trust_validated_paths)
        self.allow_external_reference = bool(allow_external_reference)
        self.records: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = REF_VIDEO_REQUIRED_METADATA - set(record)
            if missing:
                raise ValueError(f"{self.manifest}:{line_number} missing metadata {sorted(missing)}")
            if record["schema_version"] != REF_VIDEO_CACHE_SCHEMA_VERSION:
                raise ValueError(
                    f"{record['sample_id']} has schema {record['schema_version']!r}; "
                    f"expected {REF_VIDEO_CACHE_SCHEMA_VERSION!r}"
                )
            for field in ("tensor_path", "text_tensor_path"):
                record[field] = str(self._resolve_cache_path(record[field]))
            for field in ("target_video_paths", "proxy_video_paths"):
                paths = record[field]
                if not isinstance(paths, dict) or set(paths) != {"left", "right"}:
                    raise ValueError(f"{record['sample_id']} {field} must contain left/right paths")
                record[field] = {side: str(self._resolve_media_path(paths[side])) for side in ("left", "right")}
            self._validate_metadata(record)
            self.records.append(record)
        if not self.records:
            raise ValueError("The split-screen Ref2VA cache manifest is empty")
        sample_ids = [str(record["sample_id"]) for record in self.records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_id values must be unique")
        for field in (
            "vae_config_hash",
            "text_encoder_hash",
            "text_encoder_layer",
            "text_encoder_execution_revision",
            "transformer_ref_config_hash",
            "height",
            "width",
        ):
            values = {str(record[field]) for record in self.records}
            if len(values) != 1:
                raise ValueError(f"Split-screen cache mixes {field} values: {sorted(values)}")

    def _resolve_cache_path(self, path: str | Path) -> Path:
        value = Path(path)
        candidate = value if value.is_absolute() else self.allowed_root / value
        if self.trust_validated_paths:
            resolved = Path(os.path.abspath(os.fspath(candidate)))
            root = os.path.abspath(os.fspath(self.allowed_root))
            if os.path.commonpath((os.fspath(resolved), root)) != root:
                raise ValueError(f"Cache path escapes allowed_root {self.allowed_root}: {resolved}")
            return resolved
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.allowed_root)
        except ValueError as error:
            raise ValueError(f"Cache path escapes allowed_root {self.allowed_root}: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    def _resolve_media_path(self, path: str | Path) -> Path:
        value = Path(path)
        resolved = value if value.is_absolute() else self.allowed_root / value
        resolved = Path(os.path.abspath(os.fspath(resolved))) if self.trust_validated_paths else resolved.resolve()
        if not self.trust_validated_paths and not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        sample_id = str(metadata["sample_id"])
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        if metadata["reference_kinds"] != ["image", "video"] or bool(metadata["reference_has_audio"]):
            raise ValueError(f"{sample_id} must contain one image then one silent proxy-video reference")
        if not bool(metadata["reference_image_frozen"]):
            raise ValueError(f"{sample_id} reference image must be frozen")
        if metadata["reference_image_source"] == EXTERNAL_REFERENCE and self.allow_external_reference:
            validate_external_reference(metadata)
        elif metadata["reference_image_source"] != TARGET_REFERENCE:
            raise ValueError(f"{sample_id} has the wrong reference-image source")
        geometry = cache_geometry(metadata)
        split = metadata["split_screen"]
        if not isinstance(split, dict) or any(
            split.get(key) != value
            for key, value in {
                "layout": "horizontal",
                "left_agent": "agent_000",
                "right_agent": "agent_001",
                "seam_column": geometry["seam_column"],
            }.items()
        ):
            raise ValueError(f"{sample_id} has an invalid split-screen contract")
        if not isinstance(metadata["scene_description"], str):
            raise ValueError(f"{sample_id} scene_description must be a string")
        original_prompt = metadata["original_prompt"]
        if not isinstance(original_prompt, str) or not original_prompt.strip():
            raise ValueError(f"{sample_id} original_prompt must be a non-empty string")
        if str(metadata["prompt"]) != original_prompt + str(metadata["cp_prompt_suffix"]):
            raise ValueError(f"{sample_id} prompt must equal original_prompt plus cp_prompt_suffix")
        if int(metadata["text_encoder_layer"]) != 50:
            raise ValueError(f"{sample_id} must use Qwen hidden_states[50]")
        for field, expected in (
            ("num_condition_image_rows", geometry["image_rows"]),
            ("num_condition_proxy_video_rows", geometry["proxy_rows"]),
            ("num_condition_video_rows", geometry["image_rows"] + geometry["proxy_rows"]),
            ("num_target_video_rows", geometry["video_rows"]),
        ):
            if int(metadata[field]) != expected:
                raise ValueError(f"{sample_id} {field} must equal {expected}")
        if int(metadata["cache_context_parallel_degree"]) <= 0:
            raise ValueError(f"{sample_id} has an invalid cache CP degree")
        for field, expected in (
            ("vae_config_hash", self.expected_vae_hash),
            ("text_encoder_hash", self.expected_text_encoder_hash),
            ("transformer_ref_config_hash", self.expected_transformer_ref_hash),
        ):
            if expected is not None and metadata[field] != expected:
                raise ValueError(f"{sample_id} {field} mismatch")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CachedH3RefVideoSample:
        metadata = dict(self.records[index])
        geometry = cache_geometry(metadata)
        sample_id = str(metadata["sample_id"])
        main_path = Path(metadata["tensor_path"])
        text_path = Path(metadata["text_tensor_path"])
        if not self.trust_validated_paths:
            for path in (main_path, text_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
        if self.verify_sha256:
            if metadata["reference_image_source"] == EXTERNAL_REFERENCE:
                if sha256_file(Path(metadata["reference_image_path"])) != metadata["reference_image_sha256"]:
                    raise ValueError(f"{sample_id} external reference SHA mismatch")
            for path, expected in ((main_path, metadata["sha256"]), (text_path, metadata["text_sha256"])):
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(f"Cache SHA mismatch for {sample_id}: {actual} != {expected}")
            for path_field, hash_field in (
                ("target_video_paths", "target_video_sha256"),
                ("proxy_video_paths", "proxy_video_sha256"),
            ):
                for side in ("left", "right"):
                    actual = sha256_file(Path(metadata[path_field][side]))
                    expected = metadata[hash_field][side]
                    if actual != expected:
                        raise ValueError(f"{sample_id} {path_field}.{side} SHA mismatch")

        tensors = load_file(main_path)
        text_tensors = load_file(text_path)
        if set(tensors) != REF_VIDEO_MAIN_TENSORS:
            raise ValueError(f"{sample_id} main tensor keys must be {sorted(REF_VIDEO_MAIN_TENSORS)}")
        if set(text_tensors) != REF_VIDEO_TEXT_TENSORS:
            raise ValueError(f"{sample_id} text tensor keys must be {sorted(REF_VIDEO_TEXT_TENSORS)}")
        target = tensors["clean_target_video_latent"]
        image_reference = tensors["clean_reference_image_latent"]
        video_reference = tensors["clean_reference_video_latent"]
        hidden = text_tensors["encoder_hidden_states"]
        tags = text_tensors["text_token_tags"]
        for name, value, shape in (
            ("target", target, geometry["target_latent_shape"]),
            ("image reference", image_reference, geometry["image_latent_shape"]),
            ("proxy-video reference", video_reference, geometry["proxy_latent_shape"]),
        ):
            if value.dtype != torch.bfloat16 or tuple(value.shape) != shape:
                raise ValueError(f"{sample_id} {name} latent must be BF16 {list(shape)}")
            if value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{sample_id} {name} latent must be frozen and finite")
        if hidden.dtype != torch.bfloat16 or hidden.ndim != 2 or hidden.shape[1] != 5120:
            raise ValueError(f"{sample_id} encoder_hidden_states must be BF16 [L,5120]")
        if tags.dtype != torch.int64 or tags.shape != (hidden.shape[0],):
            raise ValueError(f"{sample_id} text_token_tags must be INT64 [L]")
        if not bool(((tags == 0) | (tags == 1)).all()) or not bool((tags == 0).any()) or not bool((tags == 1).any()):
            raise ValueError(f"{sample_id} presentation tags must contain both 0 and 1")
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError(f"{sample_id} contains non-finite text embeddings")

        condition_rows = geometry["image_rows"] + geometry["proxy_rows"]
        packed_rows = hidden.shape[0] + condition_rows + geometry["video_rows"]
        for field, expected in (
            ("num_text_rows", hidden.shape[0]),
            ("num_condition_image_rows", geometry["image_rows"]),
            ("num_condition_proxy_video_rows", geometry["proxy_rows"]),
            ("num_condition_video_rows", condition_rows),
            ("num_target_video_rows", geometry["video_rows"]),
            ("num_packed_rows", packed_rows),
        ):
            if int(metadata[field]) != expected:
                raise ValueError(f"{sample_id} {field} does not match cached tensors")

        sample = CachedH3RefVideoSample(target, image_reference, video_reference, hidden, tags, metadata)
        layout = sample.build_layout(context_parallel_degree=int(metadata["cache_context_parallel_degree"]))
        if layout.sequence_length != packed_rows or layout.audio_indices.numel() != 0:
            raise AssertionError(f"{sample_id} official Ref2VA layout disagrees with the cache")
        return sample


__all__ = [
    "CachedH3RefVideoDataset",
    "CachedH3RefVideoSample",
    "IMAGE_REFERENCE_LATENT_SHAPE",
    "REF_VIDEO_CACHE_SCHEMA_VERSION",
    "TARGET_LATENT_SHAPE",
    "TARGET_VIDEO_ROWS",
]
