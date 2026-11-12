"""Strict reader for offline H3 VAE/Qwen distillation caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


REQUIRED_METADATA = {
    "sample_id",
    "tensor_path",
    "prompt",
    "num_frames",
    "height",
    "width",
    "fps",
    "source",
    "license_tag",
    "vae_config_hash",
    "text_encoder_hash",
}


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedH3VideoSample:
    clean_video_latent: torch.Tensor
    encoder_hidden_states: torch.Tensor
    text_token_tags: torch.Tensor
    metadata: dict[str, Any]


class CachedH3VideoDataset:
    """Read one JSONL manifest whose samples point to H3-native safetensors files."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        expected_vae_hash: str | None = None,
        expected_text_encoder_hash: str | None = None,
        verify_sha256: bool = False,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        self.expected_vae_hash = expected_vae_hash
        self.expected_text_encoder_hash = expected_text_encoder_hash
        self.verify_sha256 = verify_sha256
        self.records = []
        for line_number, raw_line in enumerate(self.manifest.read_text().splitlines(), start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            missing = REQUIRED_METADATA - set(record)
            if missing:
                raise ValueError(f"Manifest line {line_number} is missing fields: {sorted(missing)}.")
            tensor_path = Path(record["tensor_path"])
            if not tensor_path.is_absolute():
                tensor_path = self.manifest.parent / tensor_path
            record["tensor_path"] = str(tensor_path.resolve())
            self.records.append(record)
        if not self.records:
            raise ValueError("The H3 cache manifest is empty.")
        sample_ids = [record["sample_id"] for record in self.records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_id values must be unique within an H3 cache manifest.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CachedH3VideoSample:
        metadata = dict(self.records[index])
        tensor_path = Path(metadata["tensor_path"])
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        if self.verify_sha256:
            expected = metadata.get("sha256")
            if not expected:
                raise ValueError(f"Sample {metadata['sample_id']} has no sha256 field.")
            actual = sha256_file(tensor_path)
            if actual != expected:
                raise ValueError(f"SHA256 mismatch for {tensor_path}: {actual} != {expected}.")
        if self.expected_vae_hash and metadata["vae_config_hash"] != self.expected_vae_hash:
            raise ValueError(f"VAE hash mismatch for sample {metadata['sample_id']}.")
        if self.expected_text_encoder_hash and metadata["text_encoder_hash"] != self.expected_text_encoder_hash:
            raise ValueError(f"Text encoder hash mismatch for sample {metadata['sample_id']}.")
        if not str(metadata["license_tag"]).strip():
            raise ValueError(f"Sample {metadata['sample_id']} has an empty license_tag.")

        tensors = load_file(tensor_path)
        required = {"clean_video_latent", "encoder_hidden_states", "text_token_tags"}
        missing = required - set(tensors)
        if missing:
            raise ValueError(f"Sample {metadata['sample_id']} is missing tensors: {sorted(missing)}.")
        forbidden = sorted(key for key in tensors if "audio" in key.lower() or "negative" in key.lower())
        if forbidden:
            raise ValueError(f"Video-only sample {metadata['sample_id']} contains forbidden tensors: {forbidden}.")

        latent = tensors["clean_video_latent"]
        prompt = tensors["encoder_hidden_states"]
        tags = tensors["text_token_tags"]
        if latent.ndim != 4 or latent.shape[0] != 24 or latent.dtype != torch.bfloat16:
            raise ValueError(
                f"clean_video_latent must be BF16 [24,T,H,W], got {latent.dtype} {tuple(latent.shape)}."
            )
        if prompt.ndim != 2 or prompt.shape[1] != 5120 or prompt.dtype != torch.bfloat16:
            raise ValueError(
                f"encoder_hidden_states must be BF16 [L,5120], got {prompt.dtype} {tuple(prompt.shape)}."
            )
        if tags.shape != (prompt.shape[0],) or tags.dtype != torch.int64 or not bool((tags == 1).all()):
            raise ValueError("T2V text_token_tags must be int64 [L] and contain only the text tag 1.")
        if not bool(torch.isfinite(latent).all()) or not bool(torch.isfinite(prompt).all()):
            raise ValueError(f"Sample {metadata['sample_id']} contains non-finite cached tensors.")
        num_frames = int(metadata["num_frames"])
        height, width = int(metadata["height"]), int(metadata["width"])
        expected_latent_frames = (num_frames - 5) // 17 * 5 + 2 if num_frames >= 5 and num_frames % 17 == 5 else None
        expected_shape = (expected_latent_frames, height // 16, width // 16)
        if expected_latent_frames is None or height % 16 or width % 16 or tuple(latent.shape[1:]) != expected_shape:
            raise ValueError(
                "Cached latent geometry does not match H3's 17*n+5 temporal and 16x spatial compression: "
                f"metadata=({num_frames},{metadata['height']},{metadata['width']}), latent={tuple(latent.shape)}."
            )
        if float(metadata["fps"]) != 24.0:
            raise ValueError(f"MiniMax-H3 video caches must use 24 fps, got {metadata['fps']}.")
        return CachedH3VideoSample(latent, prompt, tags, metadata)


__all__ = [
    "CachedH3VideoDataset",
    "CachedH3VideoSample",
    "sha256_file",
]
