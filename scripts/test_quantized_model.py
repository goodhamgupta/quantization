#!/usr/bin/env python3
"""Lightweight testing of quantized model (sequential loading)."""

import torch
from transformers import AutoModel, AutoProcessor
import time
import json
from pathlib import Path


def test_quantized_model(
    original_path: str = "tomoro-colqwen3-embed-4b",
    quantized_path: str = "tomoro-colqwen3-embed-4b-autoround",
):
    """Test quantized model with lightweight sequential approach."""

    print("=" * 80)
    print("ColQwen3 Quantized Model Testing")
    print("=" * 80)

    # Load processor once
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        original_path,
        trust_remote_code=True
    )

    # Test queries
    test_queries = [
        "What is shown in this document?",
        "Explain quantum computing",
        "What is machine learning?",
    ]

    # Test original model first, then unload
    print("\n=== Testing Original Model ===")
    print("Loading original model...")
    original_model = AutoModel.from_pretrained(
        original_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    original_embeddings = []
    original_times = []

    for query in test_queries:
        inputs = processor(text=query, return_tensors="pt")
        inputs = {k: v.to(original_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            start = time.time()
            outputs = original_model(**inputs)
            elapsed = time.time() - start

            original_embeddings.append(outputs.embeddings.flatten().float().cpu())
            original_times.append(elapsed)

        print(f"  Query: '{query[:40]}...' - {elapsed*1000:.1f}ms")

    # Unload original model
    del original_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("✓ Original model tested and unloaded")

    # Now test quantized model
    print("\n=== Testing Quantized Model ===")
    print("Loading quantized model...")
    quantized_model = AutoModel.from_pretrained(
        quantized_path,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    quantized_embeddings = []
    quantized_times = []

    for query in test_queries:
        inputs = processor(text=query, return_tensors="pt")
        inputs = {k: v.to(quantized_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            start = time.time()
            outputs = quantized_model(**inputs)
            elapsed = time.time() - start

            quantized_embeddings.append(outputs.embeddings.flatten().float().cpu())
            quantized_times.append(elapsed)

        print(f"  Query: '{query[:40]}...' - {elapsed*1000:.1f}ms")

    print("✓ Quantized model tested")

    # Compare embeddings
    print("\n=== Quality Comparison ===")
    cosine_sims = []

    for i, query in enumerate(test_queries):
        orig_emb = original_embeddings[i]
        quant_emb = quantized_embeddings[i]

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_emb.unsqueeze(0),
            quant_emb.unsqueeze(0)
        ).item()

        cosine_sims.append(cos_sim)

        # Check for NaN/Inf
        has_nan_inf = torch.isnan(quant_emb).any() or torch.isinf(quant_emb).any()

        print(f"\nQuery: '{query[:40]}...'")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Speedup: {original_times[i] / quantized_times[i]:.2f}x")
        print(f"  NaN/Inf: {has_nan_inf}")

    # Summary
    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    avg_speedup = sum(o/q for o, q in zip(original_times, quantized_times)) / len(original_times)

    print("\n=== Summary ===")
    print(f"Average cosine similarity: {avg_cosine:.6f}")
    print(f"Average speedup: {avg_speedup:.2f}x")

    # Quality assessment
    if avg_cosine >= 0.95:
        quality = "✓ EXCELLENT (>0.95)"
    elif avg_cosine >= 0.90:
        quality = "✓ GOOD (>0.90)"
    elif avg_cosine >= 0.85:
        quality = "⚠️  FAIR (>0.85)"
    else:
        quality = "✗ POOR (<0.85)"

    print(f"Quality: {quality}")

    # Model size comparison
    def get_dir_size(path):
        total = 0
        for file in Path(path).rglob('*'):
            if file.is_file():
                total += file.stat().st_size
        return total

    orig_size = get_dir_size(original_path) / 1e9
    quant_size = get_dir_size(quantized_path) / 1e9

    print(f"\n=== Size Comparison ===")
    print(f"Original: {orig_size:.2f} GB")
    print(f"Quantized: {quant_size:.2f} GB")
    print(f"Compression: {orig_size / quant_size:.2f}x")
    print(f"Reduction: {100 * (1 - quant_size / orig_size):.1f}%")

    # Save results
    results = {
        "avg_cosine_similarity": avg_cosine,
        "avg_speedup": avg_speedup,
        "quality": quality,
        "original_size_gb": orig_size,
        "quantized_size_gb": quant_size,
        "compression_ratio": orig_size / quant_size,
    }

    with open("test_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to test_results.json")
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test quantized ColQwen3 model")
    parser.add_argument("--original", default="tomoro-colqwen3-embed-4b", help="Original model")
    parser.add_argument("--quantized", default="tomoro-colqwen3-embed-4b-autoround", help="Quantized model")

    args = parser.parse_args()

    test_quantized_model(args.original, args.quantized)
