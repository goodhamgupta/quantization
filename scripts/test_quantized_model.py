#!/usr/bin/env python3
"""Test quantized ColQwen3 EMBEDDING VLM model with MULTIMODAL inputs (image + query)."""

import torch
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
import time
import json
from pathlib import Path


def test_quantized_model_multimodal(
    original_path: str = "tomoro-colqwen3-embed-4b",
    quantized_path: str = "tomoro-colqwen3-embed-4b-autoround",
    num_test_samples: int = 5,
):
    """
    Test quantized EMBEDDING VLM model with MULTIMODAL inputs.

    This is an EMBEDDING model - it takes image + query and produces embeddings
    for document retrieval tasks.

    Args:
        original_path: Path to original model
        quantized_path: Path to quantized model
        num_test_samples: Number of image+query pairs to test (default: 5)
    """
    print("=" * 80)
    print("ColQwen3 EMBEDDING VLM - Multimodal Quantization Test")
    print("=" * 80)
    print("\nIMPORTANT: This is an EMBEDDING VLM model")
    print("  - Input: Document image + Text query")
    print("  - Output: Multimodal embedding vector")
    print("  - Use case: Document retrieval")

    # Load processor
    print(f"\nLoading processor from {original_path}...")
    processor = AutoProcessor.from_pretrained(
        original_path,
        trust_remote_code=True
    )
    print("✓ Processor loaded")

    # Load test samples from Vidore (image + query pairs)
    print(f"\nLoading {num_test_samples} test samples from Vidore...")
    ds = load_dataset("vidore/colpali_train_set", split="train", streaming=True)
    test_samples = list(ds.take(num_test_samples))
    print(f"✓ Loaded {len(test_samples)} image+query pairs")
    for idx, sample in enumerate(test_samples):
        print(f"  [{idx}] Query: {sample['query'][:60]}... | Image size: {sample['image'].size}")

    # Test original model
    print("\n" + "=" * 80)
    print("Testing Original Model")
    print("=" * 80)
    print("Loading original model...")
    original_model = AutoModel.from_pretrained(
        original_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    print("✓ Original model loaded")

    original_embeddings = []
    original_times = []

    for idx, sample in enumerate(test_samples):
        query = sample["query"]
        image = sample["image"]  # PIL Image

        # Process BOTH query and image (multimodal input!)
        inputs = processor(
            text=query,
            images=image,  # ✅ CRITICAL: Include the document image!
            return_tensors="pt"
        )
        inputs = {k: v.to(original_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            start = time.time()
            outputs = original_model(**inputs)
            elapsed = time.time() - start

            # Get multimodal embedding
            embedding = outputs.embeddings.flatten().float().cpu()
            original_embeddings.append(embedding)
            original_times.append(elapsed)

        print(f"  [{idx}] Embedding shape: {embedding.shape} | Time: {elapsed*1000:.1f}ms")

    # Unload original model
    del original_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n✓ Original model tested and unloaded")

    # Test quantized model
    print("\n" + "=" * 80)
    print("Testing Quantized Model")
    print("=" * 80)
    print("Loading quantized model...")
    quantized_model = AutoModel.from_pretrained(
        quantized_path,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    print("✓ Quantized model loaded")

    quantized_embeddings = []
    quantized_times = []

    for idx, sample in enumerate(test_samples):
        query = sample["query"]
        image = sample["image"]  # PIL Image

        # Process BOTH query and image (multimodal input!)
        inputs = processor(
            text=query,
            images=image,  # ✅ CRITICAL: Include the document image!
            return_tensors="pt"
        )
        inputs = {k: v.to(quantized_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            start = time.time()
            outputs = quantized_model(**inputs)
            elapsed = time.time() - start

            # Get multimodal embedding
            embedding = outputs.embeddings.flatten().float().cpu()
            quantized_embeddings.append(embedding)
            quantized_times.append(elapsed)

        print(f"  [{idx}] Embedding shape: {embedding.shape} | Time: {elapsed*1000:.1f}ms")

    print("\n✓ Quantized model tested")

    # Compare multimodal embeddings
    print("\n" + "=" * 80)
    print("Multimodal Embedding Quality Comparison")
    print("=" * 80)
    cosine_sims = []

    for idx, sample in enumerate(test_samples):
        orig_emb = original_embeddings[idx]
        quant_emb = quantized_embeddings[idx]

        # Cosine similarity between multimodal embeddings
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_emb.unsqueeze(0),
            quant_emb.unsqueeze(0)
        ).item()

        cosine_sims.append(cos_sim)

        # Check for NaN/Inf
        has_nan_inf = torch.isnan(quant_emb).any() or torch.isinf(quant_emb).any()

        # L2 distance
        l2_dist = torch.norm(orig_emb - quant_emb).item()

        print(f"\n[Sample {idx}]")
        print(f"  Query: {sample['query'][:50]}...")
        print(f"  Image: {sample['image'].size}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  L2 distance: {l2_dist:.4f}")
        print(f"  Speedup: {original_times[idx] / quantized_times[idx]:.2f}x")
        print(f"  NaN/Inf detected: {has_nan_inf}")
        if has_nan_inf:
            print(f"  ⚠️  WARNING: NaN or Inf values detected!")

    # Summary statistics
    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    min_cosine = min(cosine_sims)
    max_cosine = max(cosine_sims)
    avg_speedup = sum(o/q for o, q in zip(original_times, quantized_times)) / len(original_times)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Average cosine similarity: {avg_cosine:.6f}")
    print(f"Min cosine similarity: {min_cosine:.6f}")
    print(f"Max cosine similarity: {max_cosine:.6f}")
    print(f"Average speedup: {avg_speedup:.2f}x")

    # Quality assessment
    if avg_cosine >= 0.95:
        quality = "✓ EXCELLENT (≥0.95) - Multimodal embedding quality preserved!"
    elif avg_cosine >= 0.90:
        quality = "✓ GOOD (≥0.90) - Acceptable multimodal embedding quality"
    elif avg_cosine >= 0.85:
        quality = "⚠️  FAIR (≥0.85) - Multimodal quality degraded, may affect retrieval"
    else:
        quality = "✗ POOR (<0.85) - Significant multimodal quality loss!"

    print(f"\nQuality: {quality}")

    # Model size comparison
    def get_dir_size(path):
        total = 0
        for file in Path(path).rglob('*'):
            if file.is_file():
                total += file.stat().st_size
        return total

    orig_size = get_dir_size(original_path) / 1e9
    quant_size = get_dir_size(quantized_path) / 1e9

    print(f"\n" + "=" * 80)
    print("Model Size Comparison")
    print("=" * 80)
    print(f"Original: {orig_size:.2f} GB")
    print(f"Quantized: {quant_size:.2f} GB")
    print(f"Compression: {orig_size / quant_size:.2f}x")
    print(f"Size reduction: {100 * (1 - quant_size / orig_size):.1f}%")

    # Save detailed results
    results = {
        "test_type": "multimodal_embedding",
        "num_samples": len(test_samples),
        "avg_cosine_similarity": avg_cosine,
        "min_cosine_similarity": min_cosine,
        "max_cosine_similarity": max_cosine,
        "avg_speedup": avg_speedup,
        "quality": quality,
        "original_size_gb": orig_size,
        "quantized_size_gb": quant_size,
        "compression_ratio": orig_size / quant_size,
        "sample_details": [
            {
                "sample_idx": idx,
                "query": sample["query"][:100],
                "cosine_similarity": cosine_sims[idx],
                "original_time_ms": original_times[idx] * 1000,
                "quantized_time_ms": quantized_times[idx] * 1000,
            }
            for idx, sample in enumerate(test_samples)
        ]
    }

    results_path = "multimodal_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Detailed results saved to {results_path}")

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
    print("\nNEXT STEPS:")
    print("  1. Review the quality metrics above")
    print("  2. If quality is good (>0.90), proceed with deployment")
    print("  3. For production: Run full Vidore retrieval evaluation")
    print("  4. Test on your specific use case with real documents")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test quantized ColQwen3 EMBEDDING VLM with multimodal inputs"
    )
    parser.add_argument(
        "--original",
        default="tomoro-colqwen3-embed-4b",
        help="Path to original model"
    )
    parser.add_argument(
        "--quantized",
        default="tomoro-colqwen3-embed-4b-autoround",
        help="Path to quantized model"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of image+query pairs to test (default: 5)"
    )

    args = parser.parse_args()

    test_quantized_model_multimodal(
        original_path=args.original,
        quantized_path=args.quantized,
        num_test_samples=args.num_samples
    )
