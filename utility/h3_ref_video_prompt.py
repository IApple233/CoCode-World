"""Multiplayer proxy-control prompt and Qwen system/user presentation.

Adapted to the existing image-then-video token layout from code-world-model's
src/cwm_h3_inference/presentation.py (17fa82ba2fb8168aa0d30fc51a1d1d0ebbc9180b).
The target is fully denoised here, so we do not claim that its first frame is
hard-locked or that a Retake34 prefix exists.
"""

from __future__ import annotations

# Presentation wrapping adapted from Code World Model (Apache-2.0).
# See THIRD_PARTY_NOTICES.md for the source revision and license.

import hashlib
from pathlib import Path


PRESENTATION_REVISION = "multiplayer-reference-style-natural-locomotion-system-user-v3"
SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "h3_ref_video_multiplayer_system.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
DEFAULT_TASK_PROMPT = (
    "Generate a synchronized two-view RGB video in the visual style of <Picture 1>. "
    "Show agent_000's camera on the left and agent_001's camera on the right. "
    "Preserve the rendering style, materials, colors, lighting and character identities "
    "from <Picture 1> throughout the clip, "
    "and follow the camera movement and object trajectories in <Video 1>."
)


def chat_template_parts(processor, system_prompt: str = SYSTEM_PROMPT) -> tuple[str, str]:
    """Use the actual processor template without retokenizing vision blocks."""
    marker = "__H3_MULTIPLAYER_REFERENCE_PRESENTATION__"
    formatted = processor.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if formatted.count(marker) != 1 or formatted.count(system_prompt) != 1:
        raise ValueError("System prompt and reference presentation must each occur exactly once")
    if (
        formatted.count("<|im_start|>system\n") != 1
        or formatted.count("<|im_start|>user\n") != 1
        or "<|im_start|>assistant\n" in formatted
    ):
        raise ValueError("Expected exactly one system block and one user block, without an assistant block")
    return tuple(formatted.split(marker))


def wrap_reference_presentation(tokenizer, processor, token_ids, token_tags, *, system_prompt: str = SYSTEM_PROMPT) -> tuple[list[int], list[int]]:
    """Wrap native Picture 1 / Video 1 / caption tokens, preserving modality tags."""
    if len(token_ids) != len(token_tags):
        raise ValueError("Reference token IDs and modality tags must have the same length")
    prefix, suffix = chat_template_parts(processor, system_prompt=system_prompt)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    return (
        prefix_ids + list(token_ids) + suffix_ids,
        [1] * len(prefix_ids) + list(token_tags) + [1] * len(suffix_ids),
    )
