"""Video decoding and image/video reference preparation."""

from __future__ import annotations

import numpy as np

PROFILE = {
    "num_frames": 124, "fps": 24.0, "source_height": 720, "width": 2560,
    "height": 736, "pad_top": 8, "pad_bottom": 8, "padding": "edge",
    "left_agent": "agent_000", "right_agent": "agent_001", "seam_column": 1280,
    "video_latent_shape": [24, 37, 46, 160],
    "image_latent_shape": [24, 1, 46, 160],
    "image_rows": 1840, "proxy_rows": 68080, "target_rows": 68080,
}


def decode(path, *, first_only=False):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if abs(float(stream.average_rate) - PROFILE["fps"]) > 1e-6:
            raise ValueError(f"Unexpected fps: {path}")
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if first_only:
                break
    result = np.stack(frames)
    expected = (1 if first_only else 124, 720, 2560, 3)
    if result.shape != expected:
        raise ValueError(f"{path}: {result.shape} != {expected}")
    padded = np.pad(result, ((0, 0), (8, 8), (0, 0), (0, 0)), mode="edge")
    if not np.array_equal(padded[:, 8:728], result):
        raise AssertionError("Padding changed source pixels")
    return np.ascontiguousarray(padded)


def reference_image_pixels(row):
    from cocode_world.reference import EXTERNAL_REFERENCE, TARGET_REFERENCE, load_external_reference
    if row.get("reference_image_source") == EXTERNAL_REFERENCE:
        pixels = load_external_reference(row)
        if pixels.shape != (720, 2560, 3):
            raise ValueError(f"Unexpected 720p style reference: {pixels.shape}")
        return np.pad(pixels, ((8, 8), (0, 0), (0, 0)), mode="edge")
    if row.get("reference_image_source") != TARGET_REFERENCE:
        raise ValueError("Unknown reference image source")
    return decode(row["target_video_path"], first_only=True)[0]


def references(row):
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference, MiniMaxH3VideoReference
    first = reference_image_pixels(row)
    from cocode_world.proxy import compact_references, validate_proxy_profile
    if validate_proxy_profile(row) is not None:
        return compact_references(row, first)
    proxy = decode(row["proxy_video_path"])
    setup = MiniMaxH3Ref2VASetupStep().init_pipeline()
    setup.update_components(canvas_short_edge=736, canvas_max_pixels=736 * 2560, reference_image_short_edge=736)
    normalized = setup(
        references=[MiniMaxH3ImageReference(image=first), MiniMaxH3VideoReference(frames=proxy, fps=24.0)],
        height=736, width=2560, num_frames=124, output="normalized_references",
    )
    if not np.array_equal(np.asarray(normalized[0].image), first):
        raise AssertionError("Setup changed the padded RGB anchor")
    if not np.array_equal(np.asarray(normalized[1].frames), proxy):
        raise AssertionError("Setup changed the padded proxy")
    return normalized
