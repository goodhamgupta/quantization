import argparse
import json
from pathlib import Path

LANGUAGE_MAP = {
    "eng-Latn": "English",
    "fra-Latn": "French",
    "spa-Latn": "Spanish",
    "deu-Latn": "German",
    "ita-Latn": "Italian",
    "por-Latn": "Portuguese",
}

MODEL_VARIANTS = {
    "8B": {
        "original": "results_8B_original",
        "quantized": "results_8B_w4a16_autoawq_quantized",
        "original_memory_gb": 16.7,
        "quantized_memory_gb": 7.9,
    },
    "4B": {
        "original": "results_4B_original",
        "quantized": "results_4B_w4a16_autoawq_quantized",
        "original_memory_gb": 8.4,
        "quantized_memory_gb": 3.5,
    },
}


def load_model_meta(folder: Path) -> dict:
    """Load model metadata from a folder."""
    meta_path = folder / "model_meta.json"
    with open(meta_path) as f:  # noqa: PTH123
        return json.load(f)


def load_benchmark_results(
    folder: Path, english_only: bool = False
) -> dict[str, float]:
    """Load NDCG@5 scores for all benchmarks in a folder.

    Args:
        folder: Path to the results folder
        english_only: If True, only include English language results

    Returns:
        Dictionary mapping benchmark name (with language suffix) to NDCG@5 score
    """
    results = {}
    for json_file in folder.glob("*.json"):
        if json_file.name == "model_meta.json":
            continue
        with open(json_file) as f:  # noqa: PTH123
            data = json.load(f)
        task_name = data.get("task_name", json_file.stem)

        for entry in data["scores"]["test"]:
            ndcg_at_5 = entry.get("ndcg_at_5")
            if ndcg_at_5 is None:
                continue

            languages = entry.get("languages", [])

            lang_code = languages[0]
            is_english = lang_code == "eng-Latn"
            lang_name = LANGUAGE_MAP.get(lang_code, lang_code)

            if english_only and not is_english:
                continue

            key = f"{task_name} [{lang_name}]"
            results[key] = ndcg_at_5

    return results


def count_benchmark_files(folder: Path) -> int:
    """Count the number of benchmark JSON files in a folder."""
    return len([f for f in folder.glob("*.json") if f.name != "model_meta.json"])


def generate_report(
    original_folder: Path,
    quantized_folder: Path,
    english_only: bool = False,
    model_size: str = "8B",
    original_memory_gb: float = 0.0,
    quantized_memory_gb: float = 0.0,
) -> str:
    """Generate markdown report comparing original and quantized models.

    Args:
        original_folder: Path to original model results
        quantized_folder: Path to quantized model results
        english_only: If True, only include English language results
        model_size: Model size variant (e.g., "8B", "4B")
        original_memory_gb: Memory usage of original model in GB
        quantized_memory_gb: Memory usage of quantized model in GB
    """
    original_meta = load_model_meta(original_folder)
    quantized_meta = load_model_meta(quantized_folder)

    original_file_count = count_benchmark_files(original_folder)
    quantized_file_count = count_benchmark_files(quantized_folder)

    original_results = load_benchmark_results(
        original_folder, english_only=english_only
    )
    quantized_results = load_benchmark_results(
        quantized_folder, english_only=english_only
    )

    common_benchmarks = sorted(
        set(original_results.keys()) & set(quantized_results.keys())
    )

    only_in_original = set(original_results.keys()) - set(quantized_results.keys())
    only_in_quantized = set(quantized_results.keys()) - set(original_results.keys())

    lines = []
    lang_suffix = "English Only" if english_only else "All Languages"
    lines.append(
        f"# Model Comparison Report: {model_size} Original vs Quantized ({lang_suffix})"
    )
    lines.append("")
    lines.append("## Model Information")
    lines.append("")
    lines.append("| Property | Original | Quantized |")
    lines.append("|----------|----------|-----------|")
    lines.append(
        f"| **Model Name** | {original_meta['name']} | {quantized_meta['name']} |"
    )
    lines.append(
        f"| **Parameters** | {original_meta['n_parameters'] / 1e9:.1f}B | {quantized_meta['n_parameters'] / 1e9:.1f}B |"
    )
    lines.append(
        f"| **Memory Usage** | {original_memory_gb:.1f} GB | {quantized_memory_gb:.1f} GB |"
    )
    lines.append(
        f"| **Release Date** | {original_meta['release_date']} | {quantized_meta['release_date']} |"
    )
    lines.append("")
    lines.append("## NDCG@5 Performance Comparison")
    lines.append("")
    lines.append("| Benchmark | Original | Quantized | Difference | % Change |")
    lines.append("|-----------|----------|-----------|------------|----------|")

    total_original = 0
    total_quantized = 0

    for benchmark in common_benchmarks:
        orig = original_results[benchmark]
        quant = quantized_results[benchmark]
        diff = quant - orig
        pct_change = (diff / orig) * 100 if orig != 0 else 0

        total_original += orig
        total_quantized += quant

        if diff > 0:
            change_str = f"+{pct_change:.2f}%"
        elif diff < 0:
            change_str = f"{pct_change:.2f}%"
        else:
            change_str = "0.00%"

        lines.append(
            f"| {benchmark} | {orig:.5f} | {quant:.5f} | {diff:+.5f} | {change_str} |"
        )

    n_benchmarks = len(common_benchmarks)
    if n_benchmarks > 0:
        avg_orig = total_original / n_benchmarks
        avg_quant = total_quantized / n_benchmarks
        avg_diff = avg_quant - avg_orig
        avg_pct = (avg_diff / avg_orig) * 100 if avg_orig != 0 else 0

        if avg_diff > 0:
            avg_change_str = f"+{avg_pct:.2f}%"
        elif avg_diff < 0:
            avg_change_str = f"{avg_pct:.2f}%"
        else:
            avg_change_str = "0.00%"

        lines.append("|-----------|----------|-----------|------------|----------|")
        lines.append(
            f"| **Average** | **{avg_orig:.5f}** | **{avg_quant:.5f}** | **{avg_diff:+.5f}** | **{avg_change_str}** |"
        )

    lines.append("")
    lines.append("## Summary")
    lines.append("")

    if n_benchmarks > 0:
        improved = sum(
            1 for b in common_benchmarks if quantized_results[b] > original_results[b]
        )
        degraded = sum(
            1 for b in common_benchmarks if quantized_results[b] < original_results[b]
        )
        unchanged = n_benchmarks - improved - degraded

        lines.append(f"- **Benchmark files (Original):** {original_file_count}")
        lines.append(f"- **Benchmark files (Quantized):** {quantized_file_count}")
        lines.append(f"- **Total entries evaluated:** {n_benchmarks}")
        lines.append(f"- **Entries with improvement:** {improved}")
        lines.append(f"- **Entries with degradation:** {degraded}")
        lines.append(f"- **Unchanged:** {unchanged}")

        if only_in_original:
            lines.append("")
            lines.append(
                f"**Warning:** {len(only_in_original)} entries only in original: {', '.join(sorted(only_in_original))}"
            )
        if only_in_quantized:
            lines.append("")
            lines.append(
                f"**Warning:** {len(only_in_quantized)} entries only in quantized: {', '.join(sorted(only_in_quantized))}"
            )

        lines.append("")
        lines.append("### Overall Scores")
        lines.append("")
        lines.append("| Metric | Original | Quantized | Change |")
        lines.append("|--------|----------|-----------|--------|")
        lines.append(
            f"| **Average NDCG@5** | {avg_orig:.5f} | {avg_quant:.5f} | {avg_pct:+.2f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_reports_for_variant(
    reports_dir: Path, model_size: str, verbose: bool = True
) -> bool:
    """Generate reports for a specific model variant.

    Args:
        reports_dir: Base directory containing result folders
        model_size: Model size variant (e.g., "8B", "4B")
        verbose: If True, print reports to console

    Returns:
        True if reports were generated, False if folders don't exist
    """
    if model_size not in MODEL_VARIANTS:
        print(f"Unknown model variant: {model_size}")
        return False

    config = MODEL_VARIANTS[model_size]
    original_folder = reports_dir / config["original"]
    quantized_folder = reports_dir / config["quantized"]

    # Check if folders exist
    if not original_folder.exists():
        print(f"Original folder not found: {original_folder}")
        return False
    if not quantized_folder.exists():
        print(f"Quantized folder not found: {quantized_folder}")
        return False

    print(f"\n{'=' * 80}")
    print(f"Generating reports for {model_size} model variant")
    print(f"{'=' * 80}")

    original_memory_gb = config["original_memory_gb"]
    quantized_memory_gb = config["quantized_memory_gb"]

    english_report = generate_report(
        original_folder,
        quantized_folder,
        english_only=True,
        model_size=model_size,
        original_memory_gb=original_memory_gb,
        quantized_memory_gb=quantized_memory_gb,
    )
    english_output_path = reports_dir / f"comparison_report_{model_size}_english.md"
    with open(english_output_path, "w") as f:  # noqa: PTH123
        f.write(english_report)
    print(f"English-only report generated: {english_output_path}")

    all_langs_report = generate_report(
        original_folder,
        quantized_folder,
        english_only=False,
        model_size=model_size,
        original_memory_gb=original_memory_gb,
        quantized_memory_gb=quantized_memory_gb,
    )
    all_langs_output_path = (
        reports_dir / f"comparison_report_{model_size}_all_languages.md"
    )
    with open(all_langs_output_path, "w") as f:  # noqa: PTH123
        f.write(all_langs_report)
    print(f"All languages report generated: {all_langs_output_path}")

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"{model_size} ENGLISH ONLY REPORT:")
        print(f"{'=' * 80}\n")
        print(english_report)

        print(f"\n{'=' * 80}")
        print(f"{model_size} ALL LANGUAGES REPORT:")
        print(f"{'=' * 80}\n")
        print(all_langs_report)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison reports for original vs quantized models"
    )
    parser.add_argument(
        "--variant",
        "-v",
        choices=list(MODEL_VARIANTS.keys()),
        default=None,
        help="Specific model variant to generate reports for (e.g., 8B, 4B). "
        "If not specified, generates reports for all available variants.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress printing reports to console",
    )
    args = parser.parse_args()

    reports_dir = Path(__file__).parent

    if args.variant:
        variants = [args.variant]
    else:
        variants = list(MODEL_VARIANTS.keys())

    generated_count = 0
    for variant in variants:
        if generate_reports_for_variant(reports_dir, variant, verbose=not args.quiet):
            generated_count += 1

    if generated_count == 0:
        print("\nNo reports generated. Make sure result folders exist.")
    else:
        print(f"\nGenerated reports for {generated_count} model variant(s).")


if __name__ == "__main__":
    main()
