"""MiniMax-H3-native distillation helpers."""

from .attention import (
    H3ChunkCausalMask,
    MiniMaxH3MaskedAttnProcessor,
    build_h3_chunk_causal_mask,
    use_h3_main_attention_mask,
)
from .consistency import (
    h3_clean_from_velocity,
    h3_consistency_mse,
    h3_euler_between,
    h3_interpolate,
)
from .checkpoint import load_trainable_weights, load_training_checkpoint, save_training_checkpoint
from .dcm import DcmStepOutput, h3_dcm_loss, optimize_dcm_step, sum_context_parallel_gradients
from .dfd import (
    DEFAULT_STUDENT_GRID_U,
    RNG_STREAM_NAMES,
    DfdBatchPlan,
    DfdOuterDecision,
    NamedRngStreams,
    build_dfd_teacher_input,
    make_dfd_batch_plan,
    make_dfd_microbatch_plan,
    make_dfd_outer_decision,
    mask_sequence_sha256,
    update_role_for_iteration,
)
from .dmd import (
    DmdStudentOutput,
    FakeScoreOutput,
    dmd_pseudo_loss,
    dmd_student_loss_naive,
    dmd_student_loss_replay,
    dmd_vsd_gradient,
    fake_score_dsm_loss,
    fake_score_loss,
    h3_score_perturb,
)
from .dmd_checkpoint import (
    DMD_CHECKPOINT_VERSION,
    load_dmd_training_checkpoint,
    save_dmd_training_checkpoint,
)
from .packed_video import H3VideoLayout, build_video_only_layout, patchify_video, unpatchify_video
from .rcm_adapter import H3RcmAdapter, TrainableEMA
from .role_adapters import H3DmdAdapter, H3DmdRoleManager, RoleScopedEMA
from .sampler import consistency_sample, ode_sample
from .schedules import DcmInterval, sample_dcm_interval, shifted_sigma_schedule, shift_noise_ratio

__all__ = [
    "H3ChunkCausalMask",
    "MiniMaxH3MaskedAttnProcessor",
    "build_h3_chunk_causal_mask",
    "h3_clean_from_velocity",
    "h3_consistency_mse",
    "h3_euler_between",
    "h3_interpolate",
    "DcmStepOutput",
    "DfdBatchPlan",
    "DfdOuterDecision",
    "DmdStudentOutput",
    "FakeScoreOutput",
    "H3RcmAdapter",
    "H3DmdAdapter",
    "H3DmdRoleManager",
    "H3VideoLayout",
    "RoleScopedEMA",
    "TrainableEMA",
    "DcmInterval",
    "NamedRngStreams",
    "DEFAULT_STUDENT_GRID_U",
    "RNG_STREAM_NAMES",
    "DMD_CHECKPOINT_VERSION",
    "build_dfd_teacher_input",
    "build_video_only_layout",
    "consistency_sample",
    "ode_sample",
    "patchify_video",
    "h3_dcm_loss",
    "h3_score_perturb",
    "dmd_pseudo_loss",
    "dmd_student_loss_naive",
    "dmd_student_loss_replay",
    "dmd_vsd_gradient",
    "fake_score_dsm_loss",
    "fake_score_loss",
    "load_dmd_training_checkpoint",
    "load_training_checkpoint",
    "load_trainable_weights",
    "optimize_dcm_step",
    "sample_dcm_interval",
    "make_dfd_batch_plan",
    "make_dfd_microbatch_plan",
    "make_dfd_outer_decision",
    "mask_sequence_sha256",
    "shift_noise_ratio",
    "shifted_sigma_schedule",
    "save_training_checkpoint",
    "save_dmd_training_checkpoint",
    "sum_context_parallel_gradients",
    "unpatchify_video",
    "use_h3_main_attention_mask",
    "update_role_for_iteration",
]
