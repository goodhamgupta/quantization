#!/usr/bin/env python3
"""Quantize ColQwen3 model using Auto-Round."""

from auto_round import AutoRound
from auto_round.schemes import PRESET_SCHEMES
from transformers import AutoModel, AutoProcessor
import torch
from pathlib import Path
import json


def get_available_schemes():
    """Get list of available quantization schemes."""
    return list(PRESET_SCHEMES.keys())


def quantize_colqwen3(
    model_path: str = "tomoro-colqwen3-embed-4b",
    output_dir: str = None,  # Auto-generated based on scheme if None
    scheme: str = "W4A16",
    iters: int = 200,
    nsamples: int = 8,
):
    """
    Quantize ColQwen3 with Auto-Round.

    Args:
        model_path: Path to source model
        output_dir: Output directory (auto-generated if None)
        scheme: Quantization scheme from PRESET_SCHEMES (e.g., W4A16, W2A16G32)
        iters: Number of tuning iterations
        nsamples: Number of calibration samples
    """
    # Validate scheme
    if scheme not in PRESET_SCHEMES:
        available = get_available_schemes()
        raise ValueError(f"Unknown scheme '{scheme}'. Available: {available}")

    scheme_config = PRESET_SCHEMES[scheme]
    w_bit = scheme_config.bits
    group_size = scheme_config.group_size

    # Auto-generate output directory with scheme name
    if output_dir is None:
        model_name = Path(model_path).name
        output_dir = f"{model_name}-{scheme.lower()}"

    print("=" * 80)
    print("ColQwen3 Auto-Round Quantization")
    print("=" * 80)
    print(f"\nScheme: {scheme}")
    print(f"  - Weight bits: {w_bit}")
    print(f"  - Group size: {group_size}")
    print(f"  - Symmetric: {scheme_config.sym}")
    print(f"  - Data type: {scheme_config.data_type}")
    print(f"  - Activation bits: {scheme_config.act_bits}")

    print(f"\nLoading model: {model_path}")
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
    )

    lang_modules = [n for n, _ in model.named_modules() if "language_model" in n]
    print(f"✓ Model loaded")
    print(f"  Language model modules: {len(lang_modules)}")

    print("\nLoading processor and tokenizer...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    print("✓ Processor and tokenizer loaded")

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
    print(f"Config: {scheme}, iters={iters}, nsamples={nsamples}")

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
        # Use scheme from PRESET_SCHEMES
        scheme=scheme,
        nsamples=nsamples,
        iters=iters,
        seqlen=256,
        batch_size=1,
        layer_config=layer_config,
        device_map="auto",
    )

    print("\n" + "=" * 80)
    print("Starting quantization...")
    print(f"Scheme: {scheme} (W{w_bit}A{scheme_config.act_bits})")
    print("Using text-only calibration (default: NeelNanda/pile-10k)")
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
        "scheme": scheme,
        "bits": w_bit,
        "group_size": group_size,
        "sym": scheme_config.sym,
        "data_type": scheme_config.data_type,
        "act_bits": scheme_config.act_bits,
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

    print(f"\n✓ Quantized model saved to: {output_dir}")
    print(f"  Metadata: {output_dir}/quantization_metadata.json")

    return output_dir


if __name__ == "__main__":
    import argparse

    # Get available schemes for help text
    available_schemes = get_available_schemes()

    parser = argparse.ArgumentParser(
        description="Quantize ColQwen3 with Auto-Round",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Available quantization schemes:
            W4A16     - 4-bit weights, 16-bit activations (recommended)
            W8A16     - 8-bit weights, 16-bit activations (highest quality)
            W3A16     - 3-bit weights, 16-bit activations
            W2A16     - 2-bit weights, 16-bit activations (extreme compression)
            W2A16G64  - 2-bit weights, group size 64
            W2A16G32  - 2-bit weights, group size 32 (best 2-bit quality)
            FPW8A16   - FP8 weights, 16-bit activations
            MXFP4     - MX FP4 weights and activations (requires specific hardware)
            MXFP8     - MX FP8 weights and activations (requires specific hardware)
            NVFP4     - NVIDIA FP4 (requires Blackwell/Hopper GPU)
            FP8_STATIC - Static FP8 quantization
            BF16      - BFloat16 (no quantization, baseline)
        """,
    )
    parser.add_argument(
        "--model", default="tomoro-colqwen3-embed-4b", help="Model path"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <model>-<scheme>)",
    )
    parser.add_argument(
        "--scheme",
        default="W4A16",
        choices=available_schemes,
        help="Quantization scheme (default: W4A16)",
    )
    parser.add_argument("--iters", type=int, default=200, help="Tuning iterations")
    parser.add_argument("--nsamples", type=int, default=8, help="Calibration samples")
    parser.add_argument(
        "--list-schemes",
        action="store_true",
        help="List all available quantization schemes and exit",
    )

    args = parser.parse_args()

    if args.list_schemes:
        print("Available quantization schemes:")
        for name in available_schemes:
            scheme = PRESET_SCHEMES[name]
            print(
                f"  {name:12} - {scheme.bits}-bit weights, {scheme.act_bits}-bit activations, group_size={scheme.group_size}"
            )
        exit(0)

    quantize_colqwen3(
        model_path=args.model,
        output_dir=args.output,
        scheme=args.scheme,
        iters=args.iters,
        nsamples=args.nsamples,
    )
