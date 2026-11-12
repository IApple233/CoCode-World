"""Released preset conversion, frozen fusion, inference export and resumes."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from cocode_world.decfg import (
    DEFAULT_ADAPTER_PATH, RELEASE_SHA256, USAGE,
    convert_pair, fuse_pairs, inspect_decfg_adapter, source_modules,
)


CONFIG = dict(num_attention_heads=2, attention_head_dim=4, hidden_size=6, ffn_dim=10,
              num_layers=2, num_refiner_layers=1)


def dimensions(name, config):
    inner, hidden, ffn = config["num_attention_heads"] * config["attention_head_dim"], config["hidden_size"], config["ffn_dim"]
    if name.endswith("qkv_proj"):
        return 3 * inner, hidden
    if name.endswith("out_proj"):
        return hidden, inner
    if name.endswith("fc1"):
        return 2 * ffn, hidden
    return hidden, ffn


def factor_pair(name, config=CONFIG, dtype=torch.float32):
    out_features, in_features = dimensions(name, config)
    generator = torch.Generator().manual_seed(41)
    a = torch.randn(2, in_features, generator=generator, dtype=dtype)
    b = torch.randn(out_features, 2, generator=generator, dtype=dtype)
    return a, b


def load_native_converter():
    path = Path(__file__).parents[1] / "diffusers/scripts/convert_minimax_h3_to_diffusers.py"
    spec = importlib.util.spec_from_file_location("h3_native_weight_conversion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", source_modules(CONFIG))
def test_factor_conversion_matches_native_full_weight_conversion(name):
    converter = load_native_converter()
    a, b = factor_pair(name)
    delta = b @ a
    if name.endswith("qkv_proj"):
        delta = converter.reorder_interleaved_qkv(delta, CONFIG["num_attention_heads"], CONFIG["attention_head_dim"])
    expected = dict(converter.convert_transformer_key(name + ".weight", delta, CONFIG))
    actual = {target + ".weight": up @ down for target, down, up in convert_pair(name, a, b, CONFIG)}
    assert set(actual) == set(expected)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_qkv_head_role_order_does_not_accidentally_use_comfy_order():
    a = torch.ones(1, CONFIG["hidden_size"])
    b = torch.arange(24, dtype=torch.float32).reshape(24, 1)
    pairs = convert_pair("blocks.0.attn.qkv_proj", a, b, CONFIG)
    assert pairs[0][2].flatten().tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
    assert pairs[1][2].flatten().tolist() == [4, 5, 6, 7, 16, 17, 18, 19]
    assert pairs[2][2].flatten().tolist() == [8, 9, 10, 11, 20, 21, 22, 23]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunk_fusion_matches_upstream_strength_and_keeps_base_frozen(dtype):
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(8, 12, bias=False, dtype=dtype).requires_grad_(False)
    original = model.proj.weight.detach().clone()
    a = torch.randn(3, 8).to(dtype)
    b = torch.randn(12, 3).to(dtype)
    expected = original + (b @ a) * 0.75
    assert fuse_pairs(model, [("proj", a, b)], scale=0.75, chunk_rows=4) == 1
    torch.testing.assert_close(model.proj.weight, expected, rtol=0, atol=0)
    assert not model.proj.weight.requires_grad


def tiny_h3():
    from diffusers import MiniMaxH3Transformer3DModel
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=16, hidden_size=24, num_layers=2, num_refiner_layers=1,
        ffn_dim=32, in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=8,
        freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2,
    )


def test_task_lora_checkpoint_excludes_preset_and_inference_uses_original_base(tmp_path):
    from peft import LoraConfig
    from cocode_world.ema import TrainableEMA
    from cocode_world.model import H3RefVideoTrainingModel, build_ref_video_layout
    from cocode_world.checkpoint import save_ref_video_lora_checkpoint, load_ref_video_lora_checkpoint

    torch.manual_seed(18)
    original = tiny_h3().requires_grad_(False)
    model = deepcopy(original)
    for name in source_modules(model.config):
        a, b = factor_pair(name, model.config)
        fuse_pairs(model, convert_pair(name, a * 0.02, b * 0.02, model.config))
    preset_weight = model.transformer_blocks[0].attn.to_q.weight.detach().clone()
    base_weight = original.transformer_blocks[0].attn.to_q.weight.detach().clone()
    assert not torch.equal(preset_weight, base_weight)
    lora_config = LoraConfig(r=2, lora_alpha=2, lora_dropout=0, target_modules=["to_q", "to_v"])
    model.add_adapter(lora_config, adapter_name="ref_video")
    model.enable_gradient_checkpointing()
    model.train()
    trainer = H3RefVideoTrainingModel(model)
    parameters = [p for p in trainer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=1e-3)
    ema = TrainableEMA(trainer, decay=0.9)
    layout = build_ref_video_layout(
        text_token_tags=torch.tensor([1, 0, 0, 1]), num_image_reference_latent_frames=1,
        image_reference_latent_height=4, image_reference_latent_width=4,
        num_reference_latent_frames=2, reference_latent_height=4, reference_latent_width=4,
        num_target_latent_frames=2, target_latent_height=4, target_latent_width=4,
        video_channels=4, audio_channels=6,
    )
    calls = []
    hook = model.register_forward_pre_hook(lambda *_: calls.append(1))
    result = trainer.flow_matching_loss(
        clean_target_rows=torch.randn(1, 8, 16), target_noise=torch.randn(1, 8, 16),
        clean_reference_rows=torch.randn(1, 12, 16), reference_noise=torch.randn(1, 12, 16),
        prompt_embeds=torch.randn(1, 4, 8), layout=layout, h3_timestep=0.3,
    )
    result.loss.backward(); opt.step(); ema.update(); hook.remove()
    assert len(calls) == 1 and torch.isfinite(result.loss)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    torch.testing.assert_close(model.transformer_blocks[0].attn.to_q.base_layer.weight, preset_weight, rtol=0, atol=0)
    checkpoint = save_ref_video_lora_checkpoint(tmp_path / "step-000001", model=trainer, ema=ema, optimizer=opt,
                                               step=1, metadata={"decfg_adapter_usage": USAGE})
    payload = load_file(checkpoint / "lora.safetensors")
    assert payload and all("lora_" in key and "base_layer" not in key for key in payload)
    original.add_adapter(lora_config, adapter_name="ref_video")
    inference = H3RefVideoTrainingModel(original)
    inference_ema = TrainableEMA(inference, decay=0.9)
    load_ref_video_lora_checkpoint(checkpoint, model=inference, ema=inference_ema, restore_rng=False)
    torch.testing.assert_close(original.transformer_blocks[0].attn.to_q.base_layer.weight, base_weight, rtol=0, atol=0)


def test_cli_default_off_and_checkpoint_requires_same_preset(tmp_path, monkeypatch):
    from cocode_world.training import parse_args, validate_args, checkpoint_contract, validate_resume_contract
    argv = ["train", "--manifest", "data.jsonl", "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args(); validate_args(args)
    assert args.decfg_adapter is None
    kwargs = dict(manifest_sha256="manifest", transformer_ref_hash="model", context_parallel_degree=8,
                  data_parallel_degree=4, dataset_record=dict(vae_config_hash="v", text_encoder_hash="t",
                                                             transformer_ref_config_hash="m", height=736, width=2560))
    old = checkpoint_contract(args, **kwargs)
    assert not any(key.startswith("decfg_") for key in old)
    (tmp_path / "trainer_state.json").write_text(json.dumps({"metadata": old}))
    validate_resume_contract(tmp_path, old)
    enabled = checkpoint_contract(args, **kwargs, decfg_contract={"decfg_adapter_sha256": "preset",
                                  "decfg_adapter_scale": 1.0, "decfg_adapter_usage": USAGE})
    with pytest.raises(ValueError, match="decfg_adapter"):
        validate_resume_contract(tmp_path, enabled)
    (tmp_path / "trainer_state.json").write_text(json.dumps({"metadata": enabled}))
    validate_resume_contract(tmp_path, enabled)
    for changed in [old, {**enabled, "decfg_adapter_scale": 0.5}, {**enabled, "decfg_adapter_sha256": "other"}]:
        with pytest.raises(ValueError, match="decfg_adapter"):
            validate_resume_contract(tmp_path, changed)
    monkeypatch.setattr(sys, "argv", argv + ["--preset_lora_path", "released.safetensors"])
    assert parse_args().decfg_adapter == Path("released.safetensors")


def test_cluster_environment_clears_preset_by_default_and_supports_requested_preset(tmp_path):
    from cocode_world.schedule import load_training_settings, training_environment
    settings = load_training_settings(tmp_path)
    values = dict(x.split("=", 1) for x in training_environment(settings))
    assert values["H3_REF_VIDEO_DECFG_ADAPTER"] == ""
    (tmp_path / "training_config.json").write_text(json.dumps({"decfg_adapter": str(DEFAULT_ADAPTER_PATH)}))
    settings = load_training_settings(tmp_path)
    values = dict(x.split("=", 1) for x in training_environment(settings))
    assert values["H3_REF_VIDEO_DECFG_ADAPTER"] == str(DEFAULT_ADAPTER_PATH)
    assert values["H3_REF_VIDEO_DECFG_ADAPTER_SCALE"] == "1.0"


@pytest.mark.parametrize("settings", [{"decfg_adapter": ""}, {"decfg_adapter_scale": 2},
                         {"decfg_adapter": "x", "decfg_adapter_scale": -1},
                         {"decfg_adapter": "x", "decfg_adapter_scale": float("nan")}])
def test_invalid_preset_configuration_rejected(tmp_path, settings):
    from cocode_world.schedule import load_training_settings
    (tmp_path / "training_config.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="decfg_adapter"):
        load_training_settings(tmp_path)


def test_wrong_adapter_release_is_rejected(tmp_path):
    path = tmp_path / "model_for_comfy_dit.safetensors"
    path.write_bytes(b"wrong layout or unverified adapter")
    with pytest.raises(ValueError, match="official Ref2VA"):
        inspect_decfg_adapter(path)


@pytest.mark.skipif(not DEFAULT_ADAPTER_PATH.exists(), reason="Official pretrained preset is not downloaded")
def test_real_released_adapter_all_tensors_convert_to_actual_h3_modules():
    from diffusers import MiniMaxH3Transformer3DModel
    from accelerate import init_empty_weights
    config_path = Path("models/MiniMax-H3/transformer_ref/config.json")
    cfg = json.loads(config_path.read_text())
    contract = inspect_decfg_adapter(DEFAULT_ADAPTER_PATH)
    assert contract["decfg_adapter_sha256"] == RELEASE_SHA256
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(cfg)
    modules = dict(model.named_modules())
    total = 0
    with safe_open(DEFAULT_ADAPTER_PATH, framework="pt", device="cpu") as handle:
        for name in source_modules(cfg):
            a, b = [handle.get_tensor(f"{name}.lora_{part}.default.weight") for part in ("A", "B")]
            assert a.shape[0] == b.shape[1] == 64
            for target, down, up in convert_pair(name, a, b, cfg):
                assert modules[target].weight.shape == (up.shape[0], down.shape[1])
                total += 1
    assert total == 312
