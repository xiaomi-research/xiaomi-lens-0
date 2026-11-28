# Copyright 2026 Xiaomi Corporation.
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

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoProcessor,
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3_5Config,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
    set_seed,
)

from xiaomi_lens.constants import MASK_TOKEN
from xiaomi_lens.model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration
from xiaomi_lens.train.config import (
    XiaomiLensModelConfig,
    XiaomiLensScriptArguments,
    XiaomiLensTrainingConfig,
)
from xiaomi_lens.train.preflight import build_preflight_report
from xiaomi_lens.train.trainer import XiaomiLensTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = REPO_ROOT / "examples/train/sample/sample.yaml"


@pytest.fixture
def checkpoint(tmp_path):
    special = [
        "[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
        "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", MASK_TOKEN,
    ]
    vocab = {token: idx for idx, token in enumerate(special + ["user", "assistant"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]", pad_token="[PAD]", eos_token="<|im_end|>",
        extra_special_tokens=special[2:], model_max_length=256,
    )
    processor = Qwen3VLProcessor(
        Qwen2VLImageProcessor(patch_size=16, size={"shortest_edge": 1024, "longest_edge": 65536}),
        tokenizer, Qwen3VLVideoProcessor(),
        chat_template=(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
            "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        ),
    )
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_num_key_heads": 1, "linear_num_value_heads": 2,
            "rope_parameters": {"partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]},
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
            "out_hidden_size": 32, "num_position_embeddings": 64, "patch_size": 16,
        },
        image_token_id=vocab["<|image_pad|>"], video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"], vision_end_token_id=vocab["<|vision_end|>"],
    )
    config.mask_decoder_channels = 8
    config.mask_decoder_embed_dim = 4
    model = XiaomiLensQwen3_5ForConditionalGeneration(config)
    path = tmp_path / "base"
    model.save_pretrained(path)
    processor.save_pretrained(path)
    return path, vocab[MASK_TOKEN]


def make_trainer(base, output, *, lora, max_steps, **training_kwargs):
    set_seed(42)
    return XiaomiLensTrainer(
        model=str(base),
        script_args=XiaomiLensScriptArguments(
            data_config=str(DATA_CONFIG), max_pixels=4096, model_max_length=128,
        ),
        training_args=XiaomiLensTrainingConfig(
            output_dir=str(output), use_cpu=True, max_steps=max_steps, save_steps=1,
            per_device_train_batch_size=1, learning_rate=1e-3, lr_scheduler_type="constant",
            report_to="none", lm_loss_impl="hf", disable_tqdm=True, logging_steps=1,
            **training_kwargs,
        ),
        model_args=XiaomiLensModelConfig(
            use_lora=lora, unfreeze_all_parameters=not lora,
            lora_r=2, attn_implementation="sdpa", torch_dtype="float32",
        ),
    )


def assert_same_processor(actual, expected):
    actual_processor = AutoProcessor.from_pretrained(actual)
    expected_processor = AutoProcessor.from_pretrained(expected)
    assert actual_processor.image_processor.to_dict() == expected_processor.image_processor.to_dict()
    assert actual_processor.tokenizer.model_max_length == expected_processor.tokenizer.model_max_length


def test_full_finetune_preserves_padded_vocabulary(checkpoint, tmp_path):
    base, mask_id = checkpoint
    trainer = make_trainer(base, tmp_path / "full", lora=False, max_steps=1)
    embeddings = trainer.model.get_input_embeddings().weight
    original = load_file(base / "model.safetensors")
    base_embeddings = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(base).get_input_embeddings().weight
    torch.testing.assert_close(embeddings, base_embeddings)
    assert trainer.mask_token_id == mask_id
    assert all(parameter.requires_grad for parameter in trainer.model.parameters())
    batch = next(iter(trainer.get_train_dataloader()))
    assert (batch["image_grid_thw"][:, 1:].prod(dim=-1) * 16**2 <= 4096).all()
    trainer.train()
    losses = trainer.state.log_history[0]
    assert torch.isfinite(torch.tensor([losses["lm_loss"], losses["mask_loss"]])).all()
    assert not torch.equal(embeddings, base_embeddings)
    assert not torch.equal(
        trainer.model.mask_decoder.patch_proj.weight, original["mask_decoder.patch_proj.weight"]
    )
    trainer.save_model()
    assert_same_processor(trainer.args.output_dir, base)
    model, info = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(
        trainer.args.output_dir, output_loading_info=True,
    )
    assert not info["missing_keys"]
    assert model.get_input_embeddings().num_embeddings == 32


def test_lora_resume_and_merge_preserve_segmentation(checkpoint, tmp_path, caplog):
    peft = pytest.importorskip("peft")
    base, mask_id = checkpoint
    output = tmp_path / "adapter"
    trainer = make_trainer(base, output, lora=True, max_steps=2)
    assert trainer.model_accepts_loss_kwargs
    assert trainer.mask_token_id == mask_id
    trainable = {name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
    assert any(".linear_attn.in_proj_qkv.lora_" in name for name in trainable)
    assert any(".self_attn.q_proj.lora_" in name for name in trainable)
    assert all(
        (".language_model.layers." in name and ".lora_" in name) or ".mask_decoder.modules_to_save." in name
        for name in trainable
    )
    report = build_preflight_report(
        trainer.script_args, trainer.args, trainer.model_args, trainer, {"train": trainer.train_dataset},
    )
    model_report = json.loads(report.split("[model]\n", 1)[1])
    assert model_report["backbone_class"] == "Qwen3_5Model"
    assert model_report["mask_decoder_class"] == "PixelMaskDecoder"
    modules = {item["name"]: item for item in model_report["module_parameter_summary"]}
    assert modules["visual"]["trainable_params"] == "0"
    assert modules["language_model"]["trainable_params"] != "0"
    assert modules["mask_decoder"]["trainable_ratio"] == "100.00%"
    assert modules["mask_decoder"]["total_params"] == modules["mask_decoder"]["trainable_params"]
    trainer.train()
    trainer.save_model()
    saved = load_file(output / "adapter_model.safetensors")
    original = load_file(base / "model.safetensors")
    assert any(".lora_B." in key and value.any() for key, value in saved.items())
    decoder = saved["base_model.model.mask_decoder.patch_proj.weight"]
    assert not torch.equal(decoder, original["mask_decoder.patch_proj.weight"])
    assert not (output / "model.safetensors").exists()
    assert_same_processor(output, base)
    assert_same_processor(output / "checkpoint-1", base)

    relocated = shutil.copytree(base, tmp_path / "relocated")
    resumed = make_trainer(relocated, tmp_path / "resumed", lora=True, max_steps=2)
    resumed.train(resume_from_checkpoint=str(output / "checkpoint-1"))
    assert "LoRA resume base changed" in caplog.text
    resumed.save_model()
    re_saved = load_file(Path(resumed.args.output_dir) / "adapter_model.safetensors")
    assert saved.keys() == re_saved.keys()
    for key in saved:
        torch.testing.assert_close(saved[key], re_saved[key])

    merged_path = tmp_path / "merged"
    # A previously saved adapter may contain training-only resize settings.
    adapter_processor = AutoProcessor.from_pretrained(output)
    adapter_processor.image_processor.size = {"shortest_edge": 1024, "longest_edge": 4096}
    adapter_processor.save_pretrained(output)
    subprocess.run([
        sys.executable, str(REPO_ROOT / "tools/merge_lora.py"),
        "--adapter-path", str(output), "--output-dir", str(merged_path),
        "--base-model", str(relocated),
    ], check=True)
    assert_same_processor(merged_path, base)
    assert AutoProcessor.from_pretrained(merged_path).tokenizer.convert_tokens_to_ids(MASK_TOKEN) == mask_id
    adapted = peft.PeftModel.from_pretrained(
        XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(base), output,
    ).eval()
    merged = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(merged_path).eval()
    batch = next(iter(trainer.get_train_dataloader()))
    mask_inputs = trainer._build_mask_inputs(batch)
    assert mask_inputs["mask_valid_mask"].any()
    inputs = {key: batch[key] for key in (
        "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids",
    )}
    inputs.update({key: value for key, value in mask_inputs.items() if key != "mask_targets"})
    with torch.no_grad():
        expected, actual = adapted(**inputs), merged(**inputs)
    torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual.pred_masks, expected.pred_masks, atol=1e-5, rtol=1e-4)


def test_lora_without_mask_loss_preserves_base_decoder(checkpoint, tmp_path):
    peft = pytest.importorskip("peft")
    base, _ = checkpoint
    output = tmp_path / "text-adapter"
    trainer = make_trainer(base, output, lora=True, max_steps=1, mask_loss_weight=-1.0)
    assert all(not parameter.requires_grad for parameter in trainer.model.get_base_model().mask_decoder.parameters())
    trainer.train()
    trainer.save_model()
    assert not any("mask_decoder" in key for key in load_file(output / "adapter_model.safetensors"))
    original_model = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(base)
    original_decoder = load_file(base / "model.safetensors")["mask_decoder.patch_proj.weight"]
    merged = peft.PeftModel.from_pretrained(original_model, output).merge_and_unload()
    torch.testing.assert_close(merged.mask_decoder.patch_proj.weight, original_decoder, rtol=0, atol=0)


def test_lora_checkpoint_errors(checkpoint, tmp_path):
    pytest.importorskip("peft")
    base, _ = checkpoint
    output = tmp_path / "adapter"
    trainer = make_trainer(base, output, lora=True, max_steps=1)
    trainer.save_model()
    with pytest.raises(ValueError, match="Set --model_name_or_path to a complete model"):
        make_trainer(output, tmp_path / "bad-base", lora=True, max_steps=1)
    full = make_trainer(base, tmp_path / "full", lora=False, max_steps=1)
    with pytest.raises(ValueError, match="Resume a LoRA checkpoint with --use_lora"):
        full.train(resume_from_checkpoint=str(output))

    model_path = base / "model.safetensors"
    tensors = load_file(model_path)
    del tensors["mask_decoder.patch_proj.weight"]
    save_file(tensors, model_path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="Could not load PixelMaskDecoder weights"):
        make_trainer(base, tmp_path / "missing-decoder", lora=True, max_steps=1)
