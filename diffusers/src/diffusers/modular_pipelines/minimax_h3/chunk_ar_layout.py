"""Stable modular-pipeline entry points for H3 chunk-AR row descriptors."""

from __future__ import annotations

from typing import Iterable

from utility.h3_depth_ar.causal_layout import H3ARRowBlock, build_row_metadata
from utility.h3_depth_ar.varlen_attention import (
    H3ARVarlenDescriptor,
    H3ARVarlenSegment,
    build_varlen_descriptor,
)


def build_chunk_ar_varlen_descriptor(
    blocks: Iterable[H3ARRowBlock],
) -> H3ARVarlenDescriptor:
    """Build the compact descriptor shared by training and inference."""

    return build_varlen_descriptor(build_row_metadata(tuple(blocks)))


__all__ = [
    "H3ARRowBlock",
    "H3ARVarlenDescriptor",
    "H3ARVarlenSegment",
    "build_chunk_ar_varlen_descriptor",
    "build_varlen_descriptor",
]
