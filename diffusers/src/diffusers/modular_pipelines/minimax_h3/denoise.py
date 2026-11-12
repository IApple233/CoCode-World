# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import torch

from ...models import MiniMaxH3DepthControlNetModel, MiniMaxH3Transformer3DModel
from ...schedulers import MiniMaxH3Scheduler
from ...utils import logging
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ModularPipeline
from .depth_control import depth_control_schedule


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3LoopDenoiser(ModularPipelineBlocks):
    model_name = "minimax-h3"

    def __init__(self, transformer_name: str = "transformer"):
        r"""
        Run the one MiniMax-H3 forward pass of a denoising iteration.

        Args:
            transformer_name (`str`, defaults to `"transformer"`):
                The component the forward runs against. One repository holds both checkpoint partitions —
                `transformer/` for `t2va`/`fl2va` and `transformer_ref/` for `ref2va` — under different component
                names, so which partition a loop drives is configuration, not a separate block.
        """
        self.transformer_name = transformer_name
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Runs the one MiniMax-H3 forward pass of a denoising iteration, which predicts the velocity of every row "
            "of the packed sequence at once. The checkpoint is guidance-distilled, so there is no unconditional pass "
            "and no guider."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec(self.transformer_name, MiniMaxH3Transformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The video rows of the packed sequence, conditioning rows first.",
            ),
            InputParam(
                name="audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="The channel-major audio rows of the packed sequence, reference rows first.",
            ),
            InputParam.template("prompt_embeds"),
            InputParam(
                name="row_timestep_plan",
                type_hint=list,
                required=True,
                description="One `(timestep, timestep_indices)` pair per step.",
            ),
            InputParam(
                kwargs_type="denoiser_input_fields",
                description=(
                    "The structural description of the packed sequence the transformer reads by name: `token_tags`, "
                    "`position_ids` and the three row-index tensors."
                ),
            ),
            InputParam.template("attention_kwargs"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "noise_pred",
                type_hint=torch.Tensor,
                description="Predicted velocity of the video rows of the sequence.",
            ),
            OutputParam(
                "audio_noise_pred",
                type_hint=torch.Tensor,
                description="Predicted velocity of the audio rows of the sequence.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        transformer = getattr(components, self.transformer_name)
        unique_timesteps, timestep_indices = block_state.row_timestep_plan[i]
        # The layout tags its outputs `denoiser_input_fields`, and their names are the transformer's own parameter
        # names, so the rows of the packed sequence are described to it without this block enumerating them.
        layout_kwargs = {
            name: value
            for name, value in block_state.denoiser_input_fields.items()
            if name in inspect.signature(transformer.forward).parameters
        }
        block_state.noise_pred, block_state.audio_noise_pred = transformer(
            hidden_states=block_state.latents[None],
            audio_hidden_states=block_state.audio_latents[None],
            encoder_hidden_states=block_state.prompt_embeds,
            timestep=unique_timesteps,
            timestep_indices=timestep_indices,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
            **layout_kwargs,
        )
        return components, block_state


class MiniMaxH3Ref2VALoopDenoiser(MiniMaxH3LoopDenoiser):
    model_name = "minimax-h3"

    def __init__(self):
        super().__init__(transformer_name="transformer_ref")


class MiniMaxH3Ref2VADepthLoopDenoiser(MiniMaxH3Ref2VALoopDenoiser):
    """Run one Ref2VA denoise iteration with mandatory target-aligned depth."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return super().expected_components + [
            ComponentSpec("depth_controlnet", MiniMaxH3DepthControlNetModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam(
                name="depth_rows",
                type_hint=torch.Tensor,
                required=True,
                description="Patchified H3-VAE target-only depth rows [B,N,96], cached outside the denoising loop.",
            ),
            InputParam(
                name="depth_valid",
                type_hint=torch.Tensor,
                required=True,
                description="Target-row validity mask [B,N,1].",
            ),
            InputParam(name="depth_control_scale", type_hint=float, required=False, default=1.0),
            InputParam(name="depth_control_start", type_hint=float, required=False, default=0.0),
            InputParam(name="depth_control_end", type_hint=float, required=False, default=1.0),
            InputParam(name="depth_control_schedule", type_hint=str, required=False, default="constant"),
            InputParam(
                name="num_condition_video_rows",
                type_hint=int,
                required=False,
                default=0,
                description="Leading reference rows excluded from the depth side branch.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        transformer = components.transformer_ref
        unique_timesteps, timestep_indices = block_state.row_timestep_plan[i]
        layout_kwargs = {
            name: value
            for name, value in block_state.denoiser_input_fields.items()
            if name in inspect.signature(transformer.forward).parameters
        }
        scale = depth_control_schedule(
            t,
            scale=block_state.depth_control_scale,
            start=block_state.depth_control_start,
            end=block_state.depth_control_end,
            kind=block_state.depth_control_schedule,
        )
        if scale.numel() != 1:
            raise ValueError("MiniMax-H3 depth control expects one video timestep per denoising iteration")

        num_condition_rows = block_state.num_condition_video_rows
        noisy_target_rows = block_state.latents[num_condition_rows:][None]
        depth_rows = block_state.depth_rows[None] if block_state.depth_rows.ndim == 2 else block_state.depth_rows
        depth_valid = block_state.depth_valid[None] if block_state.depth_valid.ndim == 2 else block_state.depth_valid
        video_indices = layout_kwargs["video_indices"]
        target_indices = video_indices[num_condition_rows:]
        if depth_rows.shape[:2] != noisy_target_rows.shape[:2] or target_indices.numel() != depth_rows.shape[1]:
            raise ValueError(
                "depth_rows, noisy target rows and target_video_indices must have the same row count, got "
                f"{tuple(depth_rows.shape)}, {tuple(noisy_target_rows.shape)}, {target_indices.numel()}"
            )

        if float(scale) == 0.0:
            return super().__call__(components, block_state, i, t)

        target_position_ids = layout_kwargs["position_ids"].index_select(0, target_indices)
        control_output = components.depth_controlnet(
            depth_rows=depth_rows,
            noisy_video_rows=noisy_target_rows,
            timestep=t,
            depth_valid=depth_valid,
            target_position_ids=target_position_ids,
        )
        block_state.noise_pred, block_state.audio_noise_pred = transformer(
            hidden_states=block_state.latents[None],
            audio_hidden_states=block_state.audio_latents[None],
            encoder_hidden_states=block_state.prompt_embeds,
            timestep=unique_timesteps,
            timestep_indices=timestep_indices,
            control_block_samples=control_output.control_block_samples,
            target_video_indices=target_indices,
            control_block_layers=control_output.injection_layers,
            depth_control_scale=scale,
            attention_kwargs=block_state.attention_kwargs,
            return_dict=False,
            **layout_kwargs,
        )
        return components, block_state


class MiniMaxH3LoopSchedulerStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Steps the generated video and audio rows down their own schedule. The conditioning rows are re-imposed "
            "by construction: only the generated rows are ever written, so the anchors survive the whole loop."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", MiniMaxH3Scheduler),
            ComponentSpec("audio_scheduler", MiniMaxH3Scheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="The video rows of the packed sequence, conditioning rows first.",
            ),
            InputParam(
                name="audio_latents",
                type_hint=torch.Tensor,
                required=True,
                description="The channel-major audio rows of the packed sequence, reference rows first.",
            ),
            InputParam(
                name="noise_pred",
                type_hint=torch.Tensor,
                required=True,
                description="Predicted velocity of the video rows.",
            ),
            InputParam(
                name="audio_noise_pred",
                type_hint=torch.Tensor,
                required=True,
                description="Predicted velocity of the audio rows.",
            ),
            InputParam(
                name="audio_timesteps",
                type_hint=torch.Tensor,
                required=True,
                description="Timesteps of the audio schedule.",
            ),
            InputParam(
                name="num_condition_video_rows",
                type_hint=int,
                default=0,
                description="How many leading video rows are conditioning rows.",
            ),
            InputParam(
                name="num_condition_audio_rows",
                type_hint=int,
                default=0,
                description="How many leading audio rows are reference rows.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="The video rows of the packed sequence after one step.",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="The audio rows of the packed sequence after one step.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        num_condition_video_rows = block_state.num_condition_video_rows
        num_condition_audio_rows = block_state.num_condition_audio_rows

        block_state.latents[num_condition_video_rows:] = components.scheduler.step(
            block_state.noise_pred[0, num_condition_video_rows:].float(),
            t,
            block_state.latents[num_condition_video_rows:],
            return_dict=False,
        )[0]
        block_state.audio_latents[num_condition_audio_rows:] = components.audio_scheduler.step(
            block_state.audio_noise_pred[0, num_condition_audio_rows:].float(),
            block_state.audio_timesteps[i],
            block_state.audio_latents[num_condition_audio_rows:],
            return_dict=False,
        )[0]
        return components, block_state


class MiniMaxH3DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return "Iteratively denoises the packed MiniMax-H3 sequence over the two schedules."

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", MiniMaxH3Scheduler),
            ComponentSpec("audio_scheduler", MiniMaxH3Scheduler),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam.template("timesteps", required=True, description="Timesteps of the video schedule."),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3DenoiseStep(MiniMaxH3DenoiseLoopWrapper):
    block_classes = [MiniMaxH3LoopDenoiser, MiniMaxH3LoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the `t2va` / `fl2va` MiniMax-H3 denoising loop, one forward pass per step."


class MiniMaxH3Ref2VADenoiseStep(MiniMaxH3DenoiseLoopWrapper):
    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VALoopDenoiser, MiniMaxH3LoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the `ref2va` MiniMax-H3 denoising loop, one forward pass per step."


class MiniMaxH3Ref2VADepthDenoiseStep(MiniMaxH3DenoiseLoopWrapper):
    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VADepthLoopDenoiser, MiniMaxH3LoopSchedulerStep]
    block_names = ["denoiser", "update"]

    @property
    def description(self) -> str:
        return "Runs the target-aligned depth-controlled `ref2va` denoising loop, one forward pass per step."
