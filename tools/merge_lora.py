#!/usr/bin/env python3
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

"""Export a Xiaomi-Lens-0 LoRA adapter as a standalone model checkpoint."""

import argparse
from pathlib import Path
import sys


# Make the src-layout package importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def merge_lora(adapter_path: Path, output_dir: Path, base_model: str | None = None) -> None:
    try:
        from peft import LoraConfig, PeftModel
    except ModuleNotFoundError as error:
        if error.name != "peft":
            raise
        raise ImportError('Install LoRA dependencies with `pip install -e ".[train,lora]"`.') from error
    from transformers import AutoProcessor

    from xiaomi_lens.model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration
    from xiaomi_lens.train.lora import require_mask_decoder_weights

    config = LoraConfig.from_pretrained(adapter_path)
    source = base_model or config.base_model_name_or_path
    if not source:
        raise ValueError("The adapter has no base model recorded. Pass --base-model with the original checkpoint.")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Choose a new or empty output directory: {output_dir}")

    revision = config.revision if source == config.base_model_name_or_path else None
    model, loading_info = XiaomiLensQwen3_5ForConditionalGeneration.from_pretrained(
        source,
        revision=revision,
        dtype="auto",
        output_loading_info=True,
    )
    require_mask_decoder_weights(loading_info)
    processor = AutoProcessor.from_pretrained(source, revision=model.config._commit_hash or revision)
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output_dir, max_shard_size="5GB")
    processor.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True, help="Saved LoRA adapter directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty model output directory.")
    parser.add_argument("--base-model", help="Hugging Face model ID or local copy of the original base checkpoint.")
    args = parser.parse_args()
    merge_lora(args.adapter_path, args.output_dir, args.base_model)


if __name__ == "__main__":
    main()
