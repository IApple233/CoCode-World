"""Sampling and LR continuity across epoch boundaries, CP peers and resumes."""

import json

import pytest

from utility.h3_training_schedule import (
    BufferedEpochSampler, buffered_epoch_indices, learning_rate_for_step,
    load_training_settings, training_environment,
)


@pytest.mark.parametrize("size,buffer_size", [(1, 32768), (37, 5), (37, 32768), (101, 16), (37, 0)])
def test_epoch_has_every_sample_once(size, buffer_size):
    first = list(buffered_epoch_indices(size, seed=42, epoch=0, buffer_size=buffer_size))
    second = list(buffered_epoch_indices(size, seed=42, epoch=1, buffer_size=buffer_size))
    assert sorted(first) == sorted(second) == list(range(size))
    assert first == list(buffered_epoch_indices(size, seed=42, epoch=0, buffer_size=buffer_size))
    if size > 1 and buffer_size > 1:
        assert first != second


def test_buffer_bounds_lookahead_and_seed_changes_order():
    first = list(buffered_epoch_indices(200, seed=42, epoch=0, buffer_size=8))
    assert all(index < position + 8 for position, index in enumerate(first))
    assert first != list(buffered_epoch_indices(200, seed=43, epoch=0, buffer_size=8))


def test_distributed_resume_matches_uninterrupted_stream_through_epoch_tails():
    size, replicas, accumulation = 37, 4, 2
    sampler = BufferedEpochSampler(size, seed=42, buffer_size=11)
    assignments = [sampler.step_indices(step, replicas=replicas, accumulation=accumulation) for step in range(30)]
    stream = [step[replica][microbatch] for step in assignments
              for microbatch in range(accumulation) for replica in range(replicas)]
    for epoch in range(len(stream) // size):
        assert sorted(stream[epoch * size:(epoch + 1) * size]) == list(range(size))
    for resume_step in (0, 4, 5, 11, 29):
        resumed = BufferedEpochSampler(size, seed=42, buffer_size=11)
        assert [resumed.step_indices(step, replicas=replicas, accumulation=accumulation)
                for step in range(resume_step, 30)] == assignments[resume_step:]
    cp_peers = [BufferedEpochSampler(size, seed=42, buffer_size=11) for _ in range(8)]
    assert all(peer.step_indices(4, replicas=replicas, accumulation=accumulation) == assignments[4]
               for peer in cp_peers)


def test_warmup_reaches_base_lr_then_stays_constant():
    assert learning_rate_for_step(1e-5, 0, 200) == pytest.approx(5e-8)
    assert learning_rate_for_step(1e-5, 99, 200) == pytest.approx(5e-6)
    for step in (199, 200, 500, 1000):
        assert learning_rate_for_step(1e-5, step, 200) == 1e-5
    assert learning_rate_for_step(1e-5, 0, 0) == 1e-5


def test_resumed_optimizer_and_schedule_match_continuous_training():
    import copy
    import torch

    parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)

    def update(param, opt, step):
        opt.param_groups[0]["lr"] = learning_rate_for_step(1e-5, step, 200)
        opt.zero_grad()
        (param.square().sum() * (step + 1)).backward()
        opt.step()

    for step in range(73):
        update(parameter, optimizer, step)
    saved_parameter = parameter.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    for step in range(73, 230):
        update(parameter, optimizer, step)
    resumed_parameter = torch.nn.Parameter(saved_parameter)
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=123.)
    resumed_optimizer.load_state_dict(saved_optimizer)
    for step in range(73, 230):
        update(resumed_parameter, resumed_optimizer, step)
    torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)


def test_four_node_environment_uses_the_requested_training_settings(tmp_path):
    settings = load_training_settings(tmp_path)
    environment = dict(value.split("=", 1) for value in training_environment(settings))
    assert environment["H3_REF_VIDEO_LORA_RANK"] == "64"
    assert environment["H3_REF_VIDEO_LORA_ALPHA"] == "64.0"
    assert environment["H3_REF_VIDEO_SAVE_EVERY"] == "500"
    assert environment["H3_REF_VIDEO_WARMUP_STEPS"] == "200"
    assert environment["H3_REF_VIDEO_SHUFFLE_BUFFER_SIZE"] == "32768"
    assert "H3_REF_VIDEO_SAVE_EVERY=2" in training_environment(settings, pilot=True)
    (tmp_path / "training_config.json").write_text(json.dumps({"warmup_steps": -1}))
    with pytest.raises(ValueError, match="warmup_steps"):
        load_training_settings(tmp_path)


def test_checkpoint_refuses_changed_shuffle_or_warmup_contract(tmp_path):
    from train_h3_ref_video_lora import validate_resume_contract

    metadata = {"shuffle_buffer_size": 32768, "warmup_steps": 200, "lora_rank": 64}
    (tmp_path / "trainer_state.json").write_text(json.dumps({"metadata": metadata}))
    validate_resume_contract(tmp_path, metadata)
    for name, value in (("shuffle_buffer_size", 512), ("warmup_steps", 0), ("lora_rank", 32)):
        with pytest.raises(ValueError, match=name):
            validate_resume_contract(tmp_path, {**metadata, name: value})
