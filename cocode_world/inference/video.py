"""Shared MiniMax-H3 decoder and LoRA helpers."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep

MAIN_QKVO_PATTERN = r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)"


ALL_BLOCK_LORA_PATTERN = (
    r"(transformer_blocks\.\d+|token_refiner\.refiner_blocks\.\d+)\."
    r"(attn\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))"
)


def _component_hash(model_root: Path, component: str, index_name: str) -> str:
    digest = hashlib.sha256()
    root = model_root / component
    for name in ("config.json", index_name):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_decoder(model_root: Path, device: torch.device):
    pipe = MiniMaxH3VideoDecodeStep().init_pipeline(str(model_root))
    pipe.load_components(
        names="vae",
        pretrained_model_name_or_path=str(model_root),
        dtype={"vae": torch.float32},
        device_map={"vae": str(device)},
        low_cpu_mem_usage=True,
    )
    pipe.vae.enable_tiling()
    return pipe


def _write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pixelformat="yuv420p",
    )


def _pipeline_videos(state) -> np.ndarray:
    videos = state.get("videos")
    if videos is None:
        raise RuntimeError("The H3 video decoder returned no videos output.")
    if isinstance(videos, torch.Tensor):
        videos = videos.detach().cpu().numpy()
    return np.asarray(videos)
