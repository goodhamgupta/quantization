# Model Comparison Report: 4B Original vs Quantized (English Only)

## Model Information

| Property | Original | Quantized |
|----------|----------|-----------|
| **Model Name** | TomoroAI/tomoro-colqwen3-embed-4b | TomoroAI/tomoro-colqwen3-embed-4b |
| **Parameters** | 4.0B | 4.0B |
| **Memory Usage** | 8.4 GB | 3.5 GB |
| **Release Date** | 2025-11-26 | 2025-11-26 |

## NDCG@5 Performance Comparison

| Benchmark | Original | Quantized | Difference | % Change |
|-----------|----------|-----------|------------|----------|
| Vidore2BioMedicalLecturesRetrieval [English] | 0.67177 | 0.66380 | -0.00797 | -1.19% |
| Vidore2ESGReportsHLRetrieval [English] | 0.74647 | 0.74601 | -0.00046 | -0.06% |
| Vidore2ESGReportsRetrieval [English] | 0.62995 | 0.61395 | -0.01600 | -2.54% |
| Vidore2EconomicsReportsRetrieval [English] | 0.59102 | 0.58929 | -0.00173 | -0.29% |
| Vidore3ComputerScienceRetrieval [English] | 0.74080 | 0.73873 | -0.00207 | -0.28% |
| Vidore3EnergyRetrieval [English] | 0.60348 | 0.61152 | +0.00804 | +1.33% |
| Vidore3FinanceEnRetrieval [English] | 0.67494 | 0.67323 | -0.00171 | -0.25% |
| Vidore3FinanceFrRetrieval [English] | 0.42003 | 0.42656 | +0.00653 | +1.55% |
| Vidore3HrRetrieval [English] | 0.60373 | 0.60522 | +0.00149 | +0.25% |
| Vidore3IndustrialRetrieval [English] | 0.57932 | 0.57386 | -0.00546 | -0.94% |
| Vidore3PharmaceuticalsRetrieval [English] | 0.66048 | 0.66057 | +0.00009 | +0.01% |
| Vidore3PhysicsRetrieval [English] | 0.46403 | 0.46300 | -0.00103 | -0.22% |
| VidoreArxivQARetrieval [English] | 0.90576 | 0.90194 | -0.00382 | -0.42% |
| VidoreDocVQARetrieval [English] | 0.66296 | 0.66667 | +0.00371 | +0.56% |
| VidoreInfoVQARetrieval [English] | 0.94312 | 0.94207 | -0.00105 | -0.11% |
| VidoreShiftProjectRetrieval [English] | 0.87389 | 0.87063 | -0.00326 | -0.37% |
| VidoreSyntheticDocQAAIRetrieval [English] | 0.99262 | 0.98893 | -0.00369 | -0.37% |
| VidoreSyntheticDocQAEnergyRetrieval [English] | 0.96911 | 0.96524 | -0.00387 | -0.40% |
| VidoreSyntheticDocQAGovernmentReportsRetrieval [English] | 0.97172 | 0.97172 | +0.00000 | 0.00% |
| VidoreSyntheticDocQAHealthcareIndustryRetrieval [English] | 0.99631 | 0.99262 | -0.00369 | -0.37% |
| VidoreTabfquadRetrieval [English] | 0.94329 | 0.94087 | -0.00242 | -0.26% |
| VidoreTatdqaRetrieval [English] | 0.79867 | 0.79347 | -0.00520 | -0.65% |
|-----------|----------|-----------|------------|----------|
| **Average** | **0.74743** | **0.74545** | **-0.00198** | **-0.26%** |

## Summary

- **Benchmark files (Original):** 22
- **Benchmark files (Quantized):** 22
- **Total entries evaluated:** 22
- **Entries with improvement:** 5
- **Entries with degradation:** 16
- **Unchanged:** 1

### Overall Scores

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| **Average NDCG@5** | 0.74743 | 0.74545 | -0.26% |
