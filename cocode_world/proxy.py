"""An explicitly sampled, smaller proxy reference on the original clip clock."""

from __future__ import annotations

import copy

import numpy as np


COMPACT_PROXY_PROFILE = {
    "revision": "proxy-half-resolution-stride4-source-clock-v1",
    "source_num_frames": 124,
    "source_fps": 24.0,
    "source_canvas": [720, 2560],
    "sampled_frame_indices": [*range(0, 124, 4), 123],
    "sampled_num_frames": 32,
    "content_canvas": [360, 1280],
    "canvas": [384, 1280],
    "canvas_padding": {"top": 12, "bottom": 12, "left": 0, "right": 0, "mode": "edge"},
    "resize": "per-view-area; preserve center seam",
    "vae_num_frames": 39,
    "vae_tail_padding": 7,
    "vae_tail_padding_mode": "repeat last sampled frame",
    "latent_shape": [24, 12, 24, 80],
    "time_base": "source clip seconds; padding has zero temporal extent",
}


def compact_profile() -> dict:
    return copy.deepcopy(COMPACT_PROXY_PROFILE)


def validate_proxy_profile(row: dict) -> dict | None:
    profile = row.get("proxy_reference_profile")
    if profile is not None and profile != COMPACT_PROXY_PROFILE:
        raise ValueError("Unsupported or inconsistent proxy_reference_profile")
    return profile


def proxy_latent_timestamps(row: dict) -> tuple[list[float], float] | None:
    """Map H3's (1,4,4,4,4) latent clock to sampled source-frame times."""
    profile = validate_proxy_profile(row)
    if profile is None:
        return None
    source_indices = profile["sampled_frame_indices"]
    # The seven alignment frames carry the last observed time, not new motion.
    padded = source_indices + [source_indices[-1]] * profile["vae_tail_padding"]
    offsets, cursor = [], 0
    for index in range(profile["latent_shape"][1]):
        offsets.append(padded[min(cursor, len(padded) - 1)] / profile["source_fps"])
        cursor += (1, 4, 4, 4, 4)[index % 5]
    return offsets, profile["source_num_frames"] / profile["source_fps"]


def decode_proxy(row: dict, *, for_vae: bool = False) -> np.ndarray:
    """Decode only selected RGB arrays; reduce the two views independently."""
    import av
    import cv2

    profile = validate_proxy_profile(row)
    if profile is None:
        from cocode_world.video import decode
        return decode(row["proxy_video_path"])
    wanted = set(profile["sampled_frame_indices"])
    frames = []
    with av.open(str(row["proxy_video_path"])) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if abs(float(stream.average_rate) - 24.0) > 1e-6:
            raise ValueError("Proxy source must remain 24 fps")
        count = 0
        for index, frame in enumerate(container.decode(stream)):
            count += 1
            if index not in wanted:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (720, 2560, 3):
                raise ValueError(f"Unexpected proxy source shape: {rgb.shape}")
            halves = [cv2.resize(rgb[:, side * 1280:(side + 1) * 1280], (640, 360),
                                 interpolation=cv2.INTER_AREA) for side in range(2)]
            frames.append(np.concatenate(halves, axis=1))
    if count != 124 or len(frames) != 32:
        raise ValueError(f"Incomplete proxy source/sampling: {count} source, {len(frames)} sampled")
    pixels = np.stack(frames)
    pixels = np.pad(pixels, ((0, 0), (12, 12), (0, 0), (0, 0)), mode="edge")
    if for_vae:
        pixels = np.concatenate([pixels, np.repeat(pixels[-1:], 7, axis=0)])
    return np.ascontiguousarray(pixels)


def compact_references(row: dict, first: np.ndarray):
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference, MiniMaxH3VideoReference
    profile = validate_proxy_profile(row)
    if profile is None:
        raise ValueError("compact_references needs an explicit compact proxy profile")
    # Bypass Setup's automatic 24 fps upsampling and target-canvas resizing.
    proxy = MiniMaxH3VideoReference(frames=decode_proxy(row), fps=6.0)
    proxy.source_timestamps = [i / 24.0 for i in profile["sampled_frame_indices"]]
    return [MiniMaxH3ImageReference(image=first), proxy]


def text_encoder_step(row: dict):
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    if validate_proxy_profile(row) is None:
        return MiniMaxH3Ref2VATextEncoderStep()

    class SourceClockTextEncoderStep(MiniMaxH3Ref2VATextEncoderStep):
        def _gather_vision_features(self, processor, references, fps):
            images = [ref for ref in references if ref.kind == "image"]
            vision, image_counts, _, _ = super()._gather_vision_features(processor, images, fps)
            videos = [ref for ref in references if ref.kind == "video"]
            temporal_patch = processor.video_processor.temporal_patch_size
            sampled, all_timestamps = [], []
            for reference in videos:
                times = np.asarray(reference.source_timestamps)
                # Preserve the established 2 fps conditioner view and include
                # the final observation. Never label 32 frames as 24 fps.
                grid = np.arange(0.0, times[-1] + 1e-9, 1.0 / self.video_sample_fps)
                indices = sorted({int(np.abs(times - t).argmin()) for t in grid} | {len(times) - 1})
                indices += [indices[-1]] * (-len(indices) % temporal_patch)
                sampled.append(np.stack([reference.frames[i] for i in indices]))
                all_timestamps.append([float(times[indices[j:j + temporal_patch]].mean())
                                       for j in range(0, len(indices), temporal_patch)])
            features = processor.video_processor(videos=sampled, do_sample_frames=False, return_tensors="pt")
            vision.update(pixel_values_videos=features["pixel_values_videos"], video_grid_thw=features["video_grid_thw"])
            merge = processor.image_processor.merge_size ** 2
            counts = [int(grid[1]) * int(grid[2]) // merge for grid in features["video_grid_thw"]]
            if any(int(grid[0]) != len(times) for grid, times in zip(features["video_grid_thw"], all_timestamps)):
                raise ValueError("Compact proxy vision blocks and source timestamps disagree")
            return vision, image_counts, counts, all_timestamps

    return SourceClockTextEncoderStep()
