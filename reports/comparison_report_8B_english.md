# Model Comparison Report: 8B Original vs Quantized (English Only)

## Model Information

| Property | Original | Quantized |
|----------|----------|-----------|
| **Model Name** | TomoroAI/tomoro-colqwen3-embed-8b | shubhamg2208/tomoro-ai-colqwen3-embed-8b-autoawq-w4a16 |
| **Parameters** | 8.0B | 8.0B |
| **Memory Usage** | 16724 MB | 16724 MB |
| **Release Date** | 2025-11-26 | 2025-12-12 |

## NDCG@5 Performance Comparison

| Benchmark | Original | Quantized | Difference | % Change |
|-----------|----------|-----------|------------|----------|
| Vidore2BioMedicalLecturesRetrieval [English] | 0.67838 | 0.66814 | -0.01024 | -1.51% |
| Vidore2ESGReportsHLRetrieval [English] | 0.75981 | 0.75315 | -0.00666 | -0.88% |
| Vidore2ESGReportsRetrieval [English] | 0.65488 | 0.63820 | -0.01668 | -2.55% |
| Vidore2EconomicsReportsRetrieval [English] | 0.61587 | 0.59014 | -0.02573 | -4.18% |
| Vidore3ComputerScienceRetrieval [English] | 0.74431 | 0.74155 | -0.00276 | -0.37% |
| Vidore3EnergyRetrieval [English] | 0.64907 | 0.61018 | -0.03889 | -5.99% |
| Vidore3FinanceEnRetrieval [English] | 0.68226 | 0.67471 | -0.00755 | -1.11% |
| Vidore3FinanceFrRetrieval [English] | 0.45463 | 0.42045 | -0.03418 | -7.52% |
| Vidore3HrRetrieval [English] | 0.64208 | 0.60840 | -0.03368 | -5.25% |
| Vidore3IndustrialRetrieval [English] | 0.57657 | 0.57577 | -0.00080 | -0.14% |
| Vidore3PharmaceuticalsRetrieval [English] | 0.66648 | 0.66572 | -0.00076 | -0.11% |
| Vidore3PhysicsRetrieval [English] | 0.47473 | 0.46322 | -0.01151 | -2.42% |
| VidoreArxivQARetrieval [English] | 0.91151 | 0.90922 | -0.00229 | -0.25% |
| VidoreDocVQARetrieval [English] | 0.66369 | 0.65785 | -0.00584 | -0.88% |
| VidoreInfoVQARetrieval [English] | 0.94478 | 0.94317 | -0.00161 | -0.17% |
| VidoreShiftProjectRetrieval [English] | 0.87889 | 0.87202 | -0.00687 | -0.78% |
| VidoreSyntheticDocQAAIRetrieval [English] | 0.99262 | 0.99262 | +0.00000 | 0.00% |
| VidoreSyntheticDocQAEnergyRetrieval [English] | 0.96710 | 0.96524 | -0.00186 | -0.19% |
| VidoreSyntheticDocQAGovernmentReportsRetrieval [English] | 0.97579 | 0.97172 | -0.00407 | -0.42% |
| VidoreSyntheticDocQAHealthcareIndustryRetrieval [English] | 0.99062 | 0.99262 | +0.00200 | +0.20% |
| VidoreTabfquadRetrieval [English] | 0.94231 | 0.94179 | -0.00052 | -0.06% |
| VidoreTatdqaRetrieval [English] | 0.80918 | 0.79627 | -0.01291 | -1.60% |
|-----------|----------|-----------|------------|----------|
| **Average** | **0.75798** | **0.74782** | **-0.01016** | **-1.34%** |

## Summary

- **Benchmark files (Original):** 22
- **Benchmark files (Quantized):** 22
- **Total entries evaluated:** 22
- **Entries with improvement:** 1
- **Entries with degradation:** 20
- **Unchanged:** 1

### Overall Scores

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| **Average NDCG@5** | 0.75798 | 0.74782 | -1.34% |
