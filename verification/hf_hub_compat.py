"""Compatibility shim for collecting the current diffusers tests offline.

The reused verification environment has huggingface_hub 0.36.0, while this
diffusers checkout imports two newer Hub symbols. The shim only supplies those
symbols for test collection; it never changes model weights or inference code.
"""

from __future__ import annotations

import huggingface_hub
from huggingface_hub import errors


if not hasattr(errors, "CachedRepoTreeNotFoundError"):
    class CachedRepoTreeNotFoundError(Exception):
        """Raised when a cached repository tree is unavailable."""

    errors.CachedRepoTreeNotFoundError = CachedRepoTreeNotFoundError
else:
    CachedRepoTreeNotFoundError = errors.CachedRepoTreeNotFoundError


if not hasattr(huggingface_hub, "get_cached_repo_tree"):
    def get_cached_repo_tree(*args, **kwargs):
        raise CachedRepoTreeNotFoundError(
            "get_cached_repo_tree is unavailable in the installed huggingface_hub"
        )

    huggingface_hub.get_cached_repo_tree = get_cached_repo_tree


# Transformers 4.57.6 exposes this result only through ``processor(...)``;
# the current H3 block calls the newer helper directly. Keep text-only T2VA
# compatible and preserve the conventional image/video token labels.
try:
    from transformers import Qwen3VLProcessor

    if not hasattr(Qwen3VLProcessor, "create_mm_token_type_ids"):
        def create_mm_token_type_ids(self, token_ids):
            image_id = getattr(self, "image_token_id", None)
            video_id = getattr(self, "video_token_id", None)
            result = []
            for row in token_ids:
                result.append([2 if video_id is not None and token == video_id else 1 if image_id is not None and token == image_id else 0 for token in row])
            return result

        Qwen3VLProcessor.create_mm_token_type_ids = create_mm_token_type_ids
except ImportError:
    pass
