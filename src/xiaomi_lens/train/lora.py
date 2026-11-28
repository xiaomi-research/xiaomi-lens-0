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

"""PEFT setup for fine-tuning a trained Xiaomi-Lens-0 checkpoint."""

from pathlib import Path

from torch import nn


def validate_lora_args(model_args, training_args):
    if any(value for name, value in vars(model_args).items() if name.startswith("unfreeze_")):
        raise ValueError("--use_lora cannot be combined with enabled --unfreeze_* flags.")
    if model_args.lora_r <= 0 or model_args.lora_alpha <= 0:
        raise ValueError("--lora_r and --lora_alpha must be positive.")
    if not 0 <= model_args.lora_dropout < 1:
        raise ValueError("--lora_dropout must be in [0, 1).")
    if training_args.deepspeed or training_args.fsdp:
        raise ValueError("LoRA training supports a single process or DDP. Remove --deepspeed and --fsdp.")
    if training_args.gradient_checkpointing:
        kwargs = dict(training_args.gradient_checkpointing_kwargs or {})
        kwargs.setdefault("use_reentrant", False)
        training_args.gradient_checkpointing_kwargs = kwargs


def require_mask_decoder_weights(loading_info):
    unloaded = loading_info["missing_keys"] | {key for key, *_ in loading_info["mismatched_keys"]}
    unloaded_decoder = sorted(key for key in unloaded if key.startswith("mask_decoder."))
    if unloaded_decoder:
        raise ValueError(
            "LoRA requires a trained Xiaomi-Lens-0 checkpoint. "
            f"Could not load PixelMaskDecoder weights: {', '.join(unloaded_decoder)}"
        )


def apply_lora(model, model_args, model_source, *, train_mask_decoder):
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as error:
        if error.name != "peft":
            raise
        raise ImportError('Install LoRA dependencies with `pip install -e ".[train,lora]"`.') from error

    target_modules = [
        f"model.language_model.layers.{name}"
        for name, module in model.model.language_model.layers.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if not target_modules:
        raise ValueError("No linear layers found in the Xiaomi-Lens-0 language backbone.")

    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=["mask_decoder"] if train_mask_decoder else None,
    )
    # The generic PEFT wrapper forwards Xiaomi Lens's custom segmentation inputs.
    wrapped = get_peft_model(model, config, revision=model.config._commit_hash or model_args.model_revision)
    local_source = Path(model_source)
    config.base_model_name_or_path = str(local_source.resolve()) if local_source.is_dir() else model_source
    return wrapped
