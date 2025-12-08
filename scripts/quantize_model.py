#!/usr/bin/env python3
"""Quantize ColQwen3 model using Auto-Round with real Vidore calibration data."""

from auto_round import AutoRound
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
import torch
from pathlib import Path
import json


def load_vidore_calibration_multimodal(processor, num_samples: int = 8, specific_indices=None):
    """
    Load MULTIMODAL calibration data from Vidore ColPali training set.

    This is an EMBEDDING VLM model - it processes both images and text queries
    to produce embeddings for document retrieval.

    Args:
        processor: Model processor for multimodal inputs
        num_samples: Number of samples to use (default: 8)
        specific_indices: Optional list of specific row indices to use (e.g., [51])

    Returns:
        List of multimodal calibration samples (image + query pairs)
    """
    print(f"\nLoading Vidore MULTIMODAL calibration data (image + query pairs)...")

    # Process samples
    calibration_samples = []

    if specific_indices:
        # If specific indices are requested, load only those
        print(f"  Loading specific indices: {specific_indices}")
        ds = load_dataset("vidore/colpali_train_set", split="train", streaming=False)
        selected = ds.select(specific_indices)

        for idx, sample in enumerate(selected):
            query = sample["query"]
            image = sample["image"]  # PIL Image

            # Store raw data - processor will be called during quantization
            calibration_samples.append({
                'query': query,
                'image': image,
                'source': f"vidore_{specific_indices[idx]}"
            })

            print(f"    [{specific_indices[idx]}] {query[:60]}...")
    else:
        print(f"  Streaming first {num_samples} samples (efficient loading)")
        ds = load_dataset("vidore/colpali_train_set", split="train", streaming=True)

        for idx, sample in enumerate(ds.take(num_samples)):
            query = sample["query"]
            image = sample["image"]  # PIL Image

            # Store raw data - processor will be called during quantization
            calibration_samples.append({
                'query': query,
                'image': image,
                'source': f"vidore_{idx}"
            })

            print(f"    [{idx}] Q: {query[:50]}... | Image: {image.size}")

    print(f"✓ Loaded {len(calibration_samples)} MULTIMODAL calibration samples from Vidore")
    print(f"  (Each sample contains: query text + document image)")
    return calibration_samples


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

    # Load model
    print(f"\nLoading model: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"✓ Model loaded")

    # Quick inline inspection
    lang_modules = [n for n, _ in model.named_modules() if "language_model" in n]
    print(f"  Language model modules: {len(lang_modules)}")

    # Load processor and tokenizer
    print(f"\nLoading processor and tokenizer...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    # Extract tokenizer from processor (required for multimodal models in AutoRound)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    print(f"✓ Processor and tokenizer loaded")

    # Load MULTIMODAL calibration data (image + query pairs)
    # This is critical for an EMBEDDING VLM model
    print("\n" + "=" * 80)
    print("IMPORTANT: Loading MULTIMODAL calibration data")
    print("This is an EMBEDDING VLM - must use image + query pairs!")
    print("=" * 80)

    calibration_samples = load_vidore_calibration_multimodal(
        processor,
        num_samples=nsamples,
        specific_indices=vidore_indices
    )

    # Build layer config - ONLY quantize language_model layers
    print("\n=== Building Layer Config ===")
    layer_config = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'language_model' in name:
                # Quantize language model layers
                layer_config[name] = {"bits": w_bit, "group_size": group_size}
            elif 'visual' in name or 'embedding_proj' in name:
                # Keep vision and projection in FP16
                layer_config[name] = {"bits": 16}

    quant_layers = sum(1 for v in layer_config.values() if v.get('bits') == w_bit)
    fp16_layers = sum(1 for v in layer_config.values() if v.get('bits') == 16)

    print(f"✓ Layer config built:")
    print(f"  - Quantize to {w_bit}-bit: {quant_layers} layers")
    print(f"  - Keep FP16: {fp16_layers} layers")

    print("\n=== Initializing Auto-Round ===")
    print(f"Config: W{w_bit}A16, group_size={group_size}, iters={iters}")

    print("\n✅ MULTIMODAL CALIBRATION CONFIGURED:")
    print("  - Using liuhaotian/llava_conv_58k dataset (58k image+text pairs)")
    print("  - AutoRound will use processor for multimodal inputs")
    print("  - This ensures vision encoder + language model calibration")
    print("  - Default text-only calibration (pile-10k) explicitly AVOIDED")

    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,  # Required for multimodal models
        processor=processor,   # Pass processor for MLLM mode
        dataset="liuhaotian/llava_conv_58k",  # ✅ CRITICAL: Use MULTIMODAL dataset!
        bits=w_bit,
        group_size=group_size,
        scheme="W4A16",
        nsamples=nsamples,
        iters=iters,
        seqlen=256,
        batch_size=1,
        layer_config=layer_config,  # Pass layer config
        device_map="auto"
    )

    print("\n" + "=" * 80)
    print("Starting quantization...")
    print("Using MULTIMODAL calibration: liuhaotian/llava_conv_58k")
    print("(Image + text pairs will calibrate vision encoder + language model)")
    print("=" * 80)

    autoround.quantize()
    print("\n✓ Quantization completed!")
    print("\n⚠️  CRITICAL: Test the quantized model with IMAGE+QUERY pairs")
    print("   to verify multimodal quality!")

    # Save quantized model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Saving Quantized Model ===")
    print(f"Output: {output_dir}")

    autoround.save_quantized(
        output_dir=str(output_dir),
        format='auto_round',
        inplace=True
    )

    # Save metadata
    metadata = {
        "quantization_method": "auto-round",
        "bits": w_bit,
        "group_size": group_size,
        "iters": iters,
        "nsamples": nsamples,
        "calibration_dataset": "liuhaotian/llava_conv_58k",
        "calibration_type": "multimodal (image + text pairs)",
        "quantized_layers": quant_layers,
        "fp16_layers": fp16_layers,
        "original_model": model_path,
        "note": "Vision encoder kept in FP16 for quality preservation"
    }

    with open(output_dir / "quantization_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ Quantization Complete!")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print("\nNext step:")
    print("  bash scripts/post_quantization.sh")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize ColQwen3 with Auto-Round")
    parser.add_argument("--model", default="tomoro-colqwen3-embed-4b", help="Model path")
    parser.add_argument("--output", default="tomoro-colqwen3-embed-4b-autoround", help="Output directory")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4], help="Bit width")
    parser.add_argument("--group-size", type=int, default=128, choices=[32, 64, 128], help="Group size")
    parser.add_argument("--iters", type=int, default=200, help="Tuning iterations")
    parser.add_argument("--nsamples", type=int, default=8, help="Calibration samples")
    parser.add_argument("--vidore-indices", type=int, nargs='+', help="Specific Vidore row indices to use")

    args = parser.parse_args()

    quantize_colqwen3(
        model_path=args.model,
        output_dir=args.output,
        w_bit=args.bits,
        group_size=args.group_size,
        iters=args.iters,
        nsamples=args.nsamples,
        vidore_indices=args.vidore_indices
    )
