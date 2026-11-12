#!/usr/bin/env python3
"""Download official H3, the exact Ref2VA DeCFG preset, or CoCode-World."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

BASE_REVISION = "0919fb684102ef115126612f3fbb84d1027e7317"
COCODE_REVISION = "39be9df12b0b5efc7700215eb4aba115873b3627"
DECFG_REVISION = "bdddb151d0dfb62b1dfa9e2f8ae6a6e1b179e098"
DECFG_SHA256 = "9361683349c1bf78f206c382d91b1e01b91fcddc2408e85ab39c9faa51f0d3d1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_decfg(root: Path) -> None:
    destination = root / "DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model_ref2va.safetensors"
    if destination.exists() and sha256(destination) == DECFG_SHA256:
        print(f"Verified existing DeCFG adapter: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/resolve/"
           f"{DECFG_REVISION}/model_ref2va.safetensors")
    temporary = destination.with_suffix(".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "CoCode-World/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        for block in iter(lambda: response.read(8 << 20), b""):
            stream.write(block)
    if sha256(temporary) != DECFG_SHA256:
        raise ValueError("The downloaded DeCFG file failed SHA256 verification")
    temporary.replace(destination)
    print(f"Downloaded and verified: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("base", "decfg", "cocode", "training", "all"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/step-004000"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    if args.component in ("base", "training", "all"):
        snapshot_download(
            "MiniMaxAI/MiniMax-H3", revision=BASE_REVISION,
            local_dir=args.models_dir / "MiniMax-H3",
            allow_patterns=["model_index.json", "modular_model_index.json", "LICENSE", "README.md",
                            "scheduler/*", "vae/*", "transformer_ref/*", "text_encoder/*",
                            "tokenizer/*", "processor/*"],
        )
    if args.component in ("decfg", "training", "all"):
        download_decfg(args.models_dir)
    if args.component in ("cocode", "all"):
        snapshot_download("iapple1/CoCode-World-Model", revision=COCODE_REVISION,
                          local_dir=args.checkpoint_dir)
        # The loader also validates these hashes before loading any tensor.
        import json
        state = json.loads((args.checkpoint_dir / "trainer_state.json").read_text())
        for name, record in state["payloads"].items():
            path = args.checkpoint_dir / name
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"Checkpoint verification failed: {name}")
        print("CoCode-World checkpoint verified")


if __name__ == "__main__":
    main()
