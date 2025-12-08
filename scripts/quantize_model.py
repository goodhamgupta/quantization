#!/usr/bin/env python3
"""Quantize ColQwen3 model using Auto-Round with real Vidore calibration data."""

from auto_round import AutoRound
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
import torch
from pathlib import Path
import json


def load_vidore_calibration(processor, num_samples: int = 8, specific_indices=None):
    """
    Load real calibration data from Vidore ColPali training set.

    Args:
        processor: Model processor for tokenization
        num_samples: Number of samples to use (default: 8)
        specific_indices: Optional list of specific row indices to use (e.g., [51])

    Returns:
        List of calibration samples ready for AutoRound
    """
    print(f"\nLoading Vidore calibration data...")

    # Load dataset
    ds = load_dataset("vidore/colpali_train_set", split="train", streaming=False)

    # Select samples
    if specific_indices:
        selected = ds.select(specific_indices)
        print(f"  Using specific indices: {specific_indices}")
    else:
        selected = ds.select(range(min(num_samples, len(ds))))
        print(f"  Using first {len(selected)} samples")

    # Convert to pandas for easier access
    batch = selected.to_pandas()[["question", "positive"]]

    # Process samples
    calibration_data = []
    for idx, row in batch.iterrows():
        question = row["question"]

        # Option 1: Use question only (simplest)
        text = question

        # Option 2: Include OCR text from document if available
        # Uncomment if you want to include document text:
        # if "text" in row["positive"]:
        #     doc_text = row["positive"]["text"][:500]  # Limit length
        #     text = f"{question} [DOC] {doc_text}"

        # Process through model processor
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True
        )

        calibration_data.append({
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'source': f"vidore_{idx}"
        })

        print(f"    [{idx}] {question[:60]}...")

    print(f"✓ Loaded {len(calibration_data)} calibration samples from Vidore")
    return calibration_data


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

    # Load processor
    print(f"\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print(f"✓ Processor loaded")

    # Load real Vidore calibration data
    calib_data = load_vidore_calibration(
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

    # Build dataloader for calibration
    def calib_dataloader():
        """Yield calibration samples in the format AutoRound expects."""
        device = next(model.parameters()).device
        for sample in calib_data[:nsamples]:
            yield {
                'input_ids': sample['input_ids'].unsqueeze(0).to(device),
                'attention_mask': sample['attention_mask'].unsqueeze(0).to(device)
            }

    # Convert generator to list for AutoRound
    dataloader = list(calib_dataloader())
    print(f"✓ Calibration dataloader ready with {len(dataloader)} samples")

    # Initialize Auto-Round with proper config
    print("\n=== Initializing Auto-Round ===")
    print(f"Config: W{w_bit}A16, group_size={group_size}, iters={iters}")

    try:
        autoround = AutoRound(
            model=model,
            tokenizer=None,
            bits=w_bit,
            group_size=group_size,
            scheme="asym",
            nsamples=nsamples,
            iters=iters,
            seqlen=256,
            batch_size=1,
            layer_config=layer_config,  # Pass layer config
            device=str(next(model.parameters()).device)
        )

        # Run quantization with Vidore calibration data
        print("\n" + "=" * 80)
        print("Starting quantization with Vidore calibration data...")
        print("=" * 80)

        autoround.quantize(calib_data=dataloader)
        print("\n✓ Quantization completed!")

    except Exception as e:
        print(f"\n⚠️  Standard approach failed: {e}")
        print("Trying fallback: RTN mode for VLMs (iters=0, smaller group_size)...")

        # Fallback for tricky VLM architectures
        autoround = AutoRound(
            model=model,
            tokenizer=None,
            bits=w_bit,
            group_size=32,  # Smaller for VLMs
            scheme="asym",
            nsamples=nsamples,
            iters=0,  # RTN mode (no tuning)
            seqlen=256,
            batch_size=1,
            layer_config=layer_config,
            device='cuda:0'
        )
        autoround.quantize()
        print("✓ Fallback quantization completed")

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
        "calibration_source": "vidore/colpali_train_set",
        "quantized_layers": quant_layers,
        "fp16_layers": fp16_layers,
        "original_model": model_path,
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
