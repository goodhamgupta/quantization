#!/usr/bin/env python3
"""Test quantized ColQwen3 EMBEDDING VLM model with MULTIMODAL inputs (image + query)."""

import torch
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
import time
import json
from pathlib import Path
from typing import Dict, Any, List


@torch.no_grad()
def embed_single_item(
    model,
    processor: AutoProcessor,
    item: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Compute embedding for a single item (image + text).

    The ColQwen3 model handles embedding computation internally:
    - Projects hidden states through embedding_proj_layer
    - Normalizes the output
    - Masks by attention mask
    """
    image = item.get("image")
    text = item.get("text", "")

    inputs = processor(
        images=[image] if image is not None else None,
        text=[text] if text else None,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    # outputs.embeddings shape: [1, seq_len, embed_dim]
    return outputs.embeddings[0].flatten()


@torch.no_grad()
def embed_batch(
    model,
    processor: AutoProcessor,
    items: List[Dict[str, Any]],
    device: torch.device,
) -> List[torch.Tensor]:
    """Compute embeddings for a batch of items (image + text pairs).

    The ColQwen3 model handles embedding computation internally:
    - Projects hidden states through embedding_proj_layer
    - Normalizes the output
    - Masks by attention mask

    Note: Due to variable image sizes producing different token counts,
    each item is processed individually but in a loop for cleaner code.
    True batching requires images to produce the same number of tokens.
    """
    images = []
    texts = []

    for item in items:
        image = item.get("image")
        text = item.get("text", "")
        images.append(image)
        texts.append(text)

    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    # outputs.embeddings shape: [batch_size, seq_len, embed_dim]
    embeddings = []
    for i in range(len(items)):
        embeddings.append(outputs.embeddings[i].flatten())

    return embeddings


def test_quantized_model_multimodal(
    original_path: str = "tomoro-colqwen3-embed-4b",
    quantized_path: str = "tomoro-colqwen3-embed-4b-autoround-W4A16",
    num_test_samples: int = 5,
    batch_size: int = 1,
    device: str = "cuda:0",
):
    """
    Test quantized vlm model with multimodal inputs.

    Args:
        original_path: Path to original model
        quantized_path: Path to quantized model
        num_test_samples: Number of image+query pairs to test (default: 5)
        batch_size: Batch size for processing (default: 1 for sequential)
        device: Device to run models on (default: "cuda:0")
    """
    device = torch.device(device)

    print(f"\nLoading processor from {original_path}...")
    processor = AutoProcessor.from_pretrained(original_path, trust_remote_code=True)
    print("✓ Processor loaded")

    print(f"\nLoading {num_test_samples} test samples from Vidore...")
    ds = load_dataset("vidore/colpali_train_set", split="train", streaming=True)
    test_samples = list(ds.take(num_test_samples))
    print(f"✓ Loaded {len(test_samples)} image+query pairs")
    for idx, sample in enumerate(test_samples):
        print(
            f"  [{idx}] Query: {sample['query'][:60]}... | Image size: {sample['image'].size}"
        )

    print("\n" + "=" * 80)
    print("Testing Original Model")
    print("=" * 80)
    print(f"Loading original model to {device}...")
    original_model = AutoModel.from_pretrained(
        original_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=str(device),
    ).eval()
    print("✓ Original model loaded")
    print(f"  Model device: {next(original_model.parameters()).device}")
    if device.type == "cuda":
        print(
            f"  GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
        )

    original_embeddings = []
    original_times = []

    if batch_size == 1:
        print(f"\nProcessing {len(test_samples)} samples sequentially...")
        for idx, sample in enumerate(test_samples):
            item = {"image": sample["image"], "text": sample["query"]}

            start = time.time()
            embedding = embed_single_item(original_model, processor, item, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.time() - start

            original_embeddings.append(embedding.float().cpu())
            original_times.append(elapsed)

            print(f"  [Sample {idx}] Time: {elapsed * 1000:.1f}ms")
    else:
        print(
            f"\nProcessing {len(test_samples)} samples with batch_size={batch_size}..."
        )
        for batch_start in range(0, len(test_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(test_samples))
            batch_samples = test_samples[batch_start:batch_end]
            items = [
                {"image": sample["image"], "text": sample["query"]}
                for sample in batch_samples
            ]

            start = time.time()
            batch_embeddings = embed_batch(original_model, processor, items, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.time() - start

            for emb in batch_embeddings:
                original_embeddings.append(emb.float().cpu())
            original_times.append(elapsed)

            print(
                f"  [Batch {batch_start}-{batch_end - 1}] "
                f"Time: {elapsed * 1000:.1f}ms ({elapsed * 1000 / len(items):.1f}ms/sample)"
            )

    del original_model

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("\n✓ Original model tested and unloaded")

    print("\n" + "=" * 80)
    print("Testing Quantized Model")
    print("=" * 80)
    print(f"Loading quantized model to {device}...")
    quantized_model = AutoModel.from_pretrained(
        quantized_path, trust_remote_code=True, device_map=str(device)
    ).eval()
    print("✓ Quantized model loaded")
    print(f"  Model device: {next(quantized_model.parameters()).device}")
    if device.type == "cuda":
        print(
            f"  GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
        )

    quantized_embeddings = []
    quantized_times = []

    if batch_size == 1:
        print(f"\nProcessing {len(test_samples)} samples sequentially...")
        for idx, sample in enumerate(test_samples):
            item = {"image": sample["image"], "text": sample["query"]}

            start = time.time()
            embedding = embed_single_item(quantized_model, processor, item, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.time() - start

            quantized_embeddings.append(embedding.float().cpu())
            quantized_times.append(elapsed)

            print(f"  [Sample {idx}] Time: {elapsed * 1000:.1f}ms")
    else:
        print(
            f"\nProcessing {len(test_samples)} samples with batch_size={batch_size}..."
        )
        for batch_start in range(0, len(test_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(test_samples))
            batch_samples = test_samples[batch_start:batch_end]
            items = [
                {"image": sample["image"], "text": sample["query"]}
                for sample in batch_samples
            ]

            start = time.time()
            batch_embeddings = embed_batch(quantized_model, processor, items, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.time() - start

            for emb in batch_embeddings:
                quantized_embeddings.append(emb.float().cpu())
            quantized_times.append(elapsed)

            print(
                f"  [Batch {batch_start}-{batch_end - 1}] "
                f"Time: {elapsed * 1000:.1f}ms ({elapsed * 1000 / len(items):.1f}ms/sample)"
            )

    print("\n✓ Quantized model tested")

    print("\n" + "=" * 80)
    print("Multimodal Embedding Quality Comparison")
    print("=" * 80)
    cosine_sims = []

    for idx, sample in enumerate(test_samples):
        orig_emb = original_embeddings[idx]
        quant_emb = quantized_embeddings[idx]

        # Cosine similarity between multimodal embeddings
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_emb.unsqueeze(0), quant_emb.unsqueeze(0)
        ).item()

        cosine_sims.append(cos_sim)

        has_nan_inf = torch.isnan(quant_emb).any() or torch.isinf(quant_emb).any()

        l2_dist = torch.norm(orig_emb - quant_emb).item()

        print(f"\n[Sample {idx}]")
        print(f"  Query: {sample['query'][:50]}...")
        print(f"  Image: {sample['image'].size}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  L2 distance: {l2_dist:.4f}")
        print(f"  NaN/Inf detected: {has_nan_inf}")
        if has_nan_inf:
            print("  ⚠️  WARNING: NaN or Inf values detected!")

    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    min_cosine = min(cosine_sims)
    max_cosine = max(cosine_sims)

    total_original_time = sum(original_times)
    total_quantized_time = sum(quantized_times)
    avg_speedup = (
        total_original_time / total_quantized_time if total_quantized_time > 0 else 0
    )
    avg_original_per_sample = total_original_time * 1000 / len(test_samples)
    avg_quantized_per_sample = total_quantized_time * 1000 / len(test_samples)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Average cosine similarity: {avg_cosine:.6f}")
    print(f"Min cosine similarity: {min_cosine:.6f}")
    print(f"Max cosine similarity: {max_cosine:.6f}")
    print("\nTiming:")
    print(
        f"  Original total: {total_original_time * 1000:.1f}ms ({avg_original_per_sample:.1f}ms/sample)"
    )
    print(
        f"  Quantized total: {total_quantized_time * 1000:.1f}ms ({avg_quantized_per_sample:.1f}ms/sample)"
    )
    print(f"  Speedup: {avg_speedup:.2f}x")

    if avg_cosine >= 0.95:
        quality = "✓ EXCELLENT (≥0.95) - Multimodal embedding quality preserved!"
    elif avg_cosine >= 0.90:
        quality = "✓ GOOD (≥0.90) - Acceptable multimodal embedding quality"
    elif avg_cosine >= 0.85:
        quality = "⚠️  FAIR (≥0.85) - Multimodal quality degraded, may affect retrieval"
    else:
        quality = "✗ POOR (<0.85) - Significant multimodal quality loss!"

    print(f"\nQuality: {quality}")

    def get_dir_size(path):
        total = 0
        for file in Path(path).rglob("*"):
            if file.is_file():
                total += file.stat().st_size
        return total

    orig_size = get_dir_size(original_path) / 1e9
    quant_size = get_dir_size(quantized_path) / 1e9

    print("\n" + "=" * 80)
    print("Model Size Comparison")
    print("=" * 80)
    print(f"Original: {orig_size:.2f} GB")
    print(f"Quantized: {quant_size:.2f} GB")
    print(f"Compression: {orig_size / quant_size:.2f}x")
    print(f"Size reduction: {100 * (1 - quant_size / orig_size):.1f}%")

    results = {
        "test_type": "multimodal_embedding",
        "num_samples": len(test_samples),
        "batch_size": batch_size,
        "avg_cosine_similarity": avg_cosine,
        "min_cosine_similarity": min_cosine,
        "max_cosine_similarity": max_cosine,
        "avg_speedup": avg_speedup,
        "total_original_time_ms": total_original_time * 1000,
        "total_quantized_time_ms": total_quantized_time * 1000,
        "avg_original_per_sample_ms": avg_original_per_sample,
        "avg_quantized_per_sample_ms": avg_quantized_per_sample,
        "quality": quality,
        "original_size_gb": orig_size,
        "quantized_size_gb": quant_size,
        "compression_ratio": orig_size / quant_size,
        "sample_details": [
            {
                "sample_idx": idx,
                "query": sample["query"][:100],
                "cosine_similarity": cosine_sims[idx],
            }
            for idx, sample in enumerate(test_samples)
        ],
    }

    results_path = "multimodal_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Detailed results saved to {results_path}")

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test quantized ColQwen3 EMBEDDING VLM with multimodal inputs"
    )
    parser.add_argument(
        "--original", default="tomoro-colqwen3-embed-4b", help="Path to original model"
    )
    parser.add_argument(
        "--quantized",
        default="tomoro-colqwen3-embed-4b-autoround",
        help="Path to quantized model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of image+query pairs to test (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1 for sequential)",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device to run models on (default: cuda:0)"
    )

    args = parser.parse_args()

    test_quantized_model_multimodal(
        original_path=args.original,
        quantized_path=args.quantized,
        num_test_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
    )
