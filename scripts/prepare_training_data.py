#!/usr/bin/env python3
"""Extract public CoCode-World shards and create a native H3 cache input manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utility.h3_ref_video_prompt import DEFAULT_TASK_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_SHA256


def extract_member(archive, name: str, destination: Path, expected: str) -> None:
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        return
    info = archive.getmember(name)
    if not info.isfile():
        raise ValueError(f"Expected a regular archive member: {name}")
    source = archive.extractfile(info)
    if source is None:
        raise ValueError(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    with source, temporary.open("wb") as target:
        for block in iter(lambda: source.read(8 << 20), b""):
            target.write(block)
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Dataset checksum mismatch: {name}")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Extract a prefix for a preprocessing smoke check")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    source_root, output = args.dataset_dir.resolve(), args.output_dir.resolve()
    rows = [json.loads(line) for line in (source_root / "manifests/train.jsonl").read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Training IDs must be nonempty and unique")
    output.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["shard"], []).append(row)
    prepared = []
    for shard, samples in grouped.items():
        shard_path = (source_root / shard).resolve()
        if not shard_path.is_relative_to(source_root):
            raise ValueError("Shard path escapes the dataset directory")
        with tarfile.open(shard_path, "r:") as archive:
            for row in samples:
                sid = row["sample_id"]
                if not re.fullmatch(r"[A-Za-z0-9._-]+", sid):
                    raise ValueError(f"Unsafe sample ID: {sid}")
                if (row["height"], row["width"], row["num_frames"], float(row["fps"])) != (720, 2560, 124, 24.0):
                    raise ValueError(f"Unexpected video geometry: {sid}")
                files = {"target": output / sid / "target.mp4", "proxy": output / sid / "proxy.mp4",
                         "caption_v1": output / sid / "caption.txt"}
                for key, destination in files.items():
                    extract_member(archive, row["members"][key], destination, row["sha256"][key])
                caption = files["caption_v1"].read_text().strip()
                if not caption or caption != row["caption"].strip():
                    raise ValueError(f"Caption differs from the release manifest: {sid}")
                item = dict(schema_version="gta-minimax-h3-split-screen-v1", sample_id=sid,
                            num_frames=124, fps=24.0, height=720, width=2560,
                            split_screen=row["split_screen"], source=row["source"],
                            reference_image_source="target_split_screen_frame_000000",
                            prompt=DEFAULT_TASK_PROMPT, system_prompt=SYSTEM_PROMPT,
                            system_prompt_sha256=SYSTEM_PROMPT_SHA256, scene_description=caption,
                            caption_path=str(files["caption_v1"]), caption_sha256=row["sha256"]["caption_v1"])
                for kind in ("target", "proxy"):
                    item[kind + "_video_path"] = str(files[kind])
                    item[kind + "_video_file_sha256"] = row["sha256"][kind]
                    item[kind + "_video_paths"] = {side: str(files[kind]) for side in ("left", "right")}
                    item[kind + "_video_sha256"] = {side: row["sha256"][kind] for side in ("left", "right")}
                prepared.append(item)
        print(f"Extracted {len(samples)} samples from {shard}", flush=True)
    by_id = {row["sample_id"]: row for row in prepared}
    manifest = output / "manifest.jsonl"
    temporary = manifest.with_suffix(".partial")
    temporary.write_text("".join(json.dumps(by_id[r["sample_id"]], ensure_ascii=False) + "\n" for r in rows))
    temporary.replace(manifest)
    print(f"Prepared {len(rows)} verified samples: {manifest}")


if __name__ == "__main__":
    main()
