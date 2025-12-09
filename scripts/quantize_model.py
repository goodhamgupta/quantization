#!/usr/bin/env python3
"""Quantize ColQwen3 model using Auto-Round."""

from auto_round import AutoRound
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
import torch
from pathlib import Path
import json


def quantize_colqwen3(
    model_path: str = "tomoro-colqwen3-embed-4b",
    output_dir: str = "tomoro-colqwen3-embed-4b-autoround",
    w_bit: int = 4,
    group_size: int = 128,
    iters: int = 200,
    nsamples: int = 8,
    vidore_indices=None,  # Optional: specific Vidore rows to use
):
    """
    Quantize ColQwen3 with Auto-Round using real Vidore calibration data.
    """
    print("=" * 80)
    print("ColQwen3 Auto-Round Quantization with Vidore Calibration")
    print("=" * 80)

    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    )

    lang_modules = [n for n, _ in model.named_modules() if "language_model" in n]
    print(f"  Language model modules: {len(lang_modules)}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    print("✓ Processor and tokenizer loaded")

    # Note: Calibration is handled automatically by AutoRound
    # It will use the default text-only calibration dataset (NeelNanda/pile-10k)
    # This is appropriate since we're only quantizing the language_model component

    # Build layer config - ONLY quantize language_model layers
    print("\n=== Building Layer Config ===")
    layer_config = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "language_model" in name:
                # Quantize language model layers
                layer_config[name] = {"bits": w_bit, "group_size": group_size}
            elif "visual" in name or "embedding_proj" in name:
                # Keep vision and projection in FP16
                layer_config[name] = {"bits": 16}

    quant_layers = sum(1 for v in layer_config.values() if v.get("bits") == w_bit)
    fp16_layers = sum(1 for v in layer_config.values() if v.get("bits") == 16)

    print("✓ Layer config built:")
    print(f"  - Quantize to {w_bit}-bit: {quant_layers} layers")
    print(f"  - Keep FP16: {fp16_layers} layers")

    print("\n=== Initializing Auto-Round ===")
    print(f"Config: W{w_bit}A16, group_size={group_size}, iters={iters}")

    print("\n✅ CALIBRATION CONFIGURED:")
    print("  - Using AutoRound's default text-only calibration (NeelNanda/pile-10k)")
    print("  - This is appropriate because:")
    print("    * Only quantizing language_model layers (text component)")
    print("    * Vision encoder kept in FP16 (no quantization)")
    print("  - AutoRound will use processor for MLLM mode")

    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        # Use default text-only calibration (NeelNanda/pile-10k)
        # This is appropriate since we're only quantizing the language model
        bits=w_bit,
        group_size=group_size,
        scheme="W4A16",
        nsamples=nsamples,
        iters=iters,
        seqlen=256,
        batch_size=1,
        layer_config=layer_config,
        device_map="auto",
    )

    print("\n" + "=" * 80)
    print("Starting quantization...")
    print("Using text-only calibration (default: NeelNanda/pile-10k)")
    print("(Language model will be calibrated and quantized to 4-bit)")
    print("=" * 80)

    autoround.quantize()
    print("\n✓ Quantization completed!")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Saving Quantized Model ===")
    print(f"Output: {output_dir}")

    autoround.save_quantized(
        output_dir=str(output_dir), format="auto_round", inplace=True
    )

    metadata = {
        "quantization_method": "auto-round",
        "bits": w_bit,
        "group_size": group_size,
        "iters": iters,
        "nsamples": nsamples,
        "calibration_dataset": "NeelNanda/pile-10k (AutoRound default)",
        "calibration_type": "text-only (language model only)",
        "quantized_layers": quant_layers,
        "fp16_layers": fp16_layers,
        "original_model": model_path,
        "note": "Vision encoder kept in FP16 (not quantized). Text-only calibration is appropriate since only language_model is quantized.",
    }

    with open(output_dir / "quantization_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Output: {output_dir}")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize ColQwen3 with Auto-Round")
    parser.add_argument(
        "--model", default="tomoro-colqwen3-embed-4b", help="Model path"
    )
    parser.add_argument(
        "--output",
        default="tomoro-colqwen3-embed-4b-autoround",
        help="Output directory",
    )
    parser.add_argument(
        "--bits", type=int, default=4, choices=[2, 3, 4], help="Bit width"
    )
    parser.add_argument(
        "--group-size", type=int, default=128, choices=[32, 64, 128], help="Group size"
    )
    parser.add_argument("--iters", type=int, default=200, help="Tuning iterations")
    parser.add_argument("--nsamples", type=int, default=8, help="Calibration samples")
    parser.add_argument(
        "--vidore-indices",
        type=int,
        nargs="+",
        help="Specific Vidore row indices to use",
    )

    args = parser.parse_args()

    quantize_colqwen3(
        model_path=args.model,
        output_dir=args.output,
        w_bit=args.bits,
        group_size=args.group_size,
        iters=args.iters,
        nsamples=args.nsamples,
        vidore_indices=args.vidore_indices,
    )
