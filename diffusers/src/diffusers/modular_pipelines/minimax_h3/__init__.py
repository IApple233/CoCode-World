from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["depth_control"] = [
        "MiniMaxH3DepthControlPlan",
        "depth_control_schedule",
        "h3_depth_frame_indices",
        "patchify_depth",
        "prepare_h3_depth_valid_rows",
        "prepare_h3_depth_rows",
        "resolve_depth_control_plan",
        "target_video_indices",
    ]
    _import_structure["chunk_ar_layout"] = [
        "H3ARVarlenDescriptor",
        "H3ARVarlenSegment",
        "build_chunk_ar_varlen_descriptor",
        "build_varlen_descriptor",
    ]
    _import_structure["chunk_ar_denoise"] = ["ChunkARDenoiseConfig", "denoise_chunk"]
    _import_structure["modular_blocks_minimax_h3"] = ["MiniMaxH3Blocks"]
    _import_structure["modular_pipeline"] = ["MiniMaxH3ModularPipeline"]
    _import_structure["references"] = [
        "MiniMaxH3AudioReference",
        "MiniMaxH3ImageReference",
        "MiniMaxH3Reference",
        "MiniMaxH3VideoReference",
    ]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .depth_control import (
            MiniMaxH3DepthControlPlan,
            depth_control_schedule,
            h3_depth_frame_indices,
            patchify_depth,
            prepare_h3_depth_valid_rows,
            prepare_h3_depth_rows,
            resolve_depth_control_plan,
            target_video_indices,
        )
        from .chunk_ar_layout import (
            H3ARVarlenDescriptor,
            H3ARVarlenSegment,
            build_chunk_ar_varlen_descriptor,
            build_varlen_descriptor,
        )
        from .chunk_ar_denoise import ChunkARDenoiseConfig, denoise_chunk
        from .modular_blocks_minimax_h3 import MiniMaxH3Blocks
        from .modular_pipeline import MiniMaxH3ModularPipeline
        from .references import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
            MiniMaxH3Reference,
            MiniMaxH3VideoReference,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
