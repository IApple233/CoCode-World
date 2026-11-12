#!/usr/bin/env python3
"""Verify released evaluation files, checkpoint identity, and window prompts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_video(item):
    path, expected, decode = item
    command = ["ffprobe", "-v", "error", "-threads", "2", "-select_streams", "v:0"]
    if decode:
        command += ["-count_frames"]
    command += ["-show_entries", "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,duration",
                "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    if result.stderr.strip():
        raise ValueError(f"Video decoding error in {path}: {result.stderr}")
    stream = json.loads(result.stdout)["streams"][0]
    frame_count = int(stream["nb_read_frames" if decode else "nb_frames"])
    observed = (stream["width"], stream["height"], frame_count, stream["r_frame_rate"])
    if observed != expected:
        raise ValueError(f"Unexpected video geometry: {path}: {observed}")
    return {"path": str(path.relative_to(ROOT)), "width": stream["width"], "height": stream["height"],
            "frames": frame_count, "fps": stream["r_frame_rate"], "fully_decoded": decode}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-videos", action="store_true", help="Decode every frame in all 16 videos")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = ROOT / "examples/evaluation"
    manifest = json.loads((root / "manifest.json").read_text())
    cases = manifest["cases"]
    if len(cases) != 8 or len({c["id"] for c in cases}) != 8:
        raise ValueError("The release must contain eight distinct cases")
    checked = set()
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f"Evaluation file checksum mismatch: {name}")
        checked.add(name)
    videos = []
    checkpoint = json.loads((ROOT / "provenance/checkpoint.json").read_text())
    for case in cases:
        if case["checkpoint_step"] != 4000 or case["weight_variant"] != "ema":
            raise ValueError("Wrong checkpoint in evaluation cases")
        for kind in ("ema", "lora"):
            if case[kind + "_sha256"] != checkpoint["payloads"][kind + ".safetensors"]["sha256"]:
                raise ValueError(f"Wrong checkpoint identity: {case['id']}")
        for field in ("first_frame", "proxy", "prompt", "system_prompt", "prompts", "result_preview"):
            if case[field] not in checked:
                raise ValueError(f"Unchecksummed asset: {case[field]}")
        image = (root / case["first_frame"]).read_bytes()
        if image[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", image[16:24]) != (2560, 720):
            raise ValueError("Reference image must be a complete 2560 x 720 PNG")
        if sha256(root / case["first_frame"]) != case["original_input_reference_sha256"]:
            raise ValueError("First-frame input differs from recorded inference")
        if sha256(root / case["proxy"]) != case["original_input_proxy_sha256"]:
            raise ValueError("Proxy input differs from recorded inference")
        prompt_data = json.loads((root / case["prompts"]).read_text())
        if prompt_data["seed"] != case["seed"] or len(prompt_data["windows"]) != 17:
            raise ValueError("Wrong seed or number of prompts")
        for i, window in enumerate(prompt_data["windows"]):
            start, end = i * 85, min(i * 85 + 124, 1440)
            if (window["index"], window["start_frame"], window["end_frame_exclusive"],
                window["valid_source_frames"], window["right_pad_frames"]) != (i, start, end, end-start, 124-end+start):
                raise ValueError("Prompt window timing does not match the recorded rollout")
            if window["user_prompt"] != window["task_prompt"].strip() + "\nScene description: " + window["scene_description"].strip():
                raise ValueError("Prompt text does not match its task and scene description")
            for field in ("user_prompt", "system_prompt"):
                if hashlib.sha256(window[field].encode()).hexdigest() != window[field + "_sha256"]:
                    raise ValueError("Prompt checksum mismatch")
        if (root / case["prompt"]).read_text().strip() != prompt_data["windows"][0]["user_prompt"]:
            raise ValueError("First-window prompt differs")
        videos += [(root / case["proxy"], (2560, 720, 1440, "24/1"), args.decode_videos),
                   (root / case["result_preview"], (1600, 450, 1440, "24/1"), args.decode_videos)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        media = list(executor.map(inspect_video, videos))
    report = {"status": "PASS", "cases": 8, "prompt_windows": 136, "checksummed_files": len(checked),
              "input_identity_verified": True, "videos": media,
              "fresh_full_model_training_performed": False, "fresh_full_model_inference_performed": False}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: 8 cases, 136 prompt windows, {len(checked)} checksummed files, 16 videos"
          + (" fully decoded" if args.decode_videos else " checked by metadata"))


if __name__ == "__main__":
    main()
