#!/usr/bin/env python3
"""Prepare the released 60-second cases using their original window prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_windows(source: Path, folder: Path) -> list[Path]:
    paths = [folder / f"window_{i:02d}.mp4" for i in range(17)]
    marker = folder / "windows.json"
    source_hash = sha256(source)
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["source_sha256"] == source_hash and all(
            p.is_file() and sha256(p) == old["sha256"][p.name] for p in paths
        ):
            return paths
        raise ValueError(f"Stale window files in {folder}; use a fresh output directory")
    folder.mkdir(parents=True, exist_ok=True)
    filters = ["[0:v]split=17" + "".join(f"[s{i}]" for i in range(17))]
    for i in range(17):
        start, end = 85 * i, min(85 * i + 124, 1440)
        op = f"[s{i}]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
        if end - start < 124:
            op += f",tpad=stop_mode=clone:stop={124 - (end - start)}"
        filters.append(op + f"[o{i}]")
    command = ["ffmpeg", "-y", "-v", "error", "-vsync", "0", "-threads", "2", "-i", str(source),
               "-filter_complex_threads", "4", "-filter_complex", ";".join(filters)]
    for i, path in enumerate(paths):
        command += ["-map", f"[o{i}]", "-an", "-frames:v", "124", "-c:v", "libx264", "-threads", "2",
                    "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path.with_suffix(".partial.mp4"))]
    subprocess.run(command, check=True)
    for path in paths:
        temporary = path.with_suffix(".partial.mp4")
        info = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", str(temporary),
        ]))["streams"][0]
        if (info["width"], info["height"], int(info["nb_read_frames"]), info["r_frame_rate"]) != (2560, 720, 124, "24/1"):
            raise ValueError(f"Invalid proxy window: {temporary}: {info}")
        temporary.replace(path)
    marker.write_text(json.dumps({"source_sha256": source_hash,
                                 "sha256": {p.name: sha256(p) for p in paths}}, indent=2) + "\n")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=ROOT / "examples/evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", help="Select one or more case_XX IDs; default: all eight")
    args = parser.parse_args()
    examples, output = args.examples.resolve(), args.output_dir.resolve()
    cases = json.loads((examples / "manifest.json").read_text())["cases"]
    selected = set(args.case_id or [c["id"] for c in cases])
    if not selected <= {c["id"] for c in cases}:
        parser.error("Unknown case ID")
    rows, plan_cases = [], []
    for case in cases:
        if case["id"] not in selected:
            continue
        sid = case["id"]
        reference, proxy = examples / case["first_frame"], examples / case["proxy"]
        if sha256(reference) != case["original_input_reference_sha256"] or sha256(proxy) != case["original_input_proxy_sha256"]:
            raise ValueError(f"Released inputs changed: {sid}")
        paths = make_windows(proxy, output / "media" / sid)
        prompts = json.loads((examples / case["prompts"]).read_text())
        if prompts["seed"] != case["seed"] or len(prompts["windows"]) != 17:
            raise ValueError(f"Wrong prompt schedule: {sid}")
        for i, (window, path) in enumerate(zip(prompts["windows"], paths)):
            if window["index"] != i or window["start_frame"] != i * 85:
                raise ValueError(f"Wrong window order: {sid}")
            caption = output / "captions" / sid / f"window_{i:02d}.txt"
            caption.parent.mkdir(parents=True, exist_ok=True)
            caption.write_text(window["scene_description"] + "\n")
            row = dict(schema_version="gta-minimax-h3-split-screen-v1", sample_id=f"{sid}--w{i:02d}",
                       base_sample_id=sid, num_frames=124, fps=24.0, height=720, width=2560,
                       source_canvas=[720, 2560],
                       split_screen=dict(layout="horizontal", left_agent="agent_000", right_agent="agent_001",
                                         width=2560, height=720, seam_column=1280),
                       reference_image_source="external_benchmark_image", reference_image_path=str(reference),
                       reference_image_sha256=case["original_input_reference_sha256"],
                       caption_path=str(caption), caption_sha256=sha256(caption),
                       scene_description=window["scene_description"], prompt=window["task_prompt"],
                       system_prompt=window["system_prompt"], system_prompt_sha256=window["system_prompt_sha256"],
                       eval_seeds=[case["seed"]], proxy_version="555", window_index=i, window_count=17,
                       inference_target_placeholder=True)
            # This cache schema includes a target slot. During inference its clean
            # tensor is ignored: denoising starts from case-seeded random noise.
            # Reuse the proxy for that unused slot; no ground-truth RGB is required.
            for kind in ("target", "proxy"):
                row[kind + "_video_path"] = str(path)
                row[kind + "_video_file_sha256"] = sha256(path)
                row[kind + "_video_paths"] = {side: str(path) for side in ("left", "right")}
                row[kind + "_video_sha256"] = {side: row[kind + "_video_file_sha256"] for side in ("left", "right")}
            rows.append(row)
        plan_cases.append(dict(sample_id=sid, eval_seeds=[case["seed"]]))
        print(f"Prepared {sid}: 17 verified windows", flush=True)
    checkpoint = json.loads((ROOT / "configs/step-004000.json").read_text())
    payloads = json.loads((ROOT / "provenance/checkpoint.json").read_text())["payloads"]
    plan = dict(schema_version="gta-h3-action100-proxy555-latent39-full60s-v1", proxy_version="555",
                window_frames=124, overlap_frames=39, stride_frames=85, fixed_prefix_latents=12,
                window_count=17, case_count=len(plan_cases), output_frames=1440,
                rows=[r["sample_id"] for r in rows], cases=plan_cases,
                training_manifest_sha256=checkpoint["dataset_manifest_sha256"],
                checkpoint_payloads={name: info["sha256"] for name, info in payloads.items()})
    (output / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Prepared {len(plan_cases)} cases in {output}")


if __name__ == "__main__":
    main()
