# Model Comparison Report: 4B Original vs Quantized (All Languages)

## Model Information

| Property | Original | Quantized |
|----------|----------|-----------|
| **Model Name** | TomoroAI/tomoro-colqwen3-embed-4b | shubhamg2208/tomoro-ai-colqwen3-embed-4b-autoawq-w4a16 |
| **Parameters** | 4.0B | 4.0B |
| **Memory Usage** | 8466 MB | 8466 MB |
| **Release Date** | 2025-11-26 | 2025-12-12 |

## NDCG@5 Performance Comparison

| Benchmark | Original | Quantized | Difference | % Change |
|-----------|----------|-----------|------------|----------|
| Vidore2BioMedicalLecturesRetrieval [English] | 0.67177 | 0.66380 | -0.00797 | -1.19% |
| Vidore2BioMedicalLecturesRetrieval [French] | 0.63440 | 0.63388 | -0.00052 | -0.08% |
| Vidore2BioMedicalLecturesRetrieval [German] | 0.65138 | 0.65516 | +0.00378 | +0.58% |
| Vidore2BioMedicalLecturesRetrieval [Spanish] | 0.65776 | 0.64496 | -0.01280 | -1.95% |
| Vidore2ESGReportsHLRetrieval [English] | 0.74647 | 0.74601 | -0.00046 | -0.06% |
| Vidore2ESGReportsRetrieval [English] | 0.62995 | 0.61395 | -0.01600 | -2.54% |
| Vidore2ESGReportsRetrieval [French] | 0.61270 | 0.60925 | -0.00345 | -0.56% |
| Vidore2ESGReportsRetrieval [German] | 0.60262 | 0.60913 | +0.00651 | +1.08% |
| Vidore2ESGReportsRetrieval [Spanish] | 0.65238 | 0.61384 | -0.03854 | -5.91% |
| Vidore2EconomicsReportsRetrieval [English] | 0.59102 | 0.58929 | -0.00173 | -0.29% |
| Vidore2EconomicsReportsRetrieval [French] | 0.53182 | 0.52112 | -0.01070 | -2.01% |
| Vidore2EconomicsReportsRetrieval [German] | 0.55849 | 0.54328 | -0.01521 | -2.72% |
| Vidore2EconomicsReportsRetrieval [Spanish] | 0.57064 | 0.53907 | -0.03157 | -5.53% |
| Vidore3ComputerScienceRetrieval [English] | 0.74080 | 0.73873 | -0.00207 | -0.28% |
| Vidore3ComputerScienceRetrieval [French] | 0.71492 | 0.71421 | -0.00071 | -0.10% |
| Vidore3ComputerScienceRetrieval [German] | 0.72097 | 0.71384 | -0.00713 | -0.99% |
| Vidore3ComputerScienceRetrieval [Italian] | 0.72938 | 0.71683 | -0.01255 | -1.72% |
| Vidore3ComputerScienceRetrieval [Portuguese] | 0.72535 | 0.72835 | +0.00300 | +0.41% |
| Vidore3ComputerScienceRetrieval [Spanish] | 0.71571 | 0.71948 | +0.00377 | +0.53% |
| Vidore3EnergyRetrieval [English] | 0.60348 | 0.61152 | +0.00804 | +1.33% |
| Vidore3EnergyRetrieval [French] | 0.65566 | 0.64844 | -0.00722 | -1.10% |
| Vidore3EnergyRetrieval [German] | 0.62173 | 0.62064 | -0.00109 | -0.18% |
| Vidore3EnergyRetrieval [Italian] | 0.64283 | 0.64215 | -0.00068 | -0.11% |
| Vidore3EnergyRetrieval [Portuguese] | 0.63236 | 0.62828 | -0.00408 | -0.65% |
| Vidore3EnergyRetrieval [Spanish] | 0.63481 | 0.63888 | +0.00407 | +0.64% |
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
| **Average** | **0.70023** | **0.69611** | **-0.00411** | **-0.59%** |

## Summary

- **Benchmark files (Original):** 22
- **Benchmark files (Quantized):** 22
- **Total entries evaluated:** 41
- **Entries with improvement:** 10
- **Entries with degradation:** 30
- **Unchanged:** 1

**Warning:** 30 entries only in original: Vidore3FinanceEnRetrieval [French], Vidore3FinanceEnRetrieval [German], Vidore3FinanceEnRetrieval [Italian], Vidore3FinanceEnRetrieval [Portuguese], Vidore3FinanceEnRetrieval [Spanish], Vidore3FinanceFrRetrieval [French], Vidore3FinanceFrRetrieval [German], Vidore3FinanceFrRetrieval [Italian], Vidore3FinanceFrRetrieval [Portuguese], Vidore3FinanceFrRetrieval [Spanish], Vidore3HrRetrieval [French], Vidore3HrRetrieval [German], Vidore3HrRetrieval [Italian], Vidore3HrRetrieval [Portuguese], Vidore3HrRetrieval [Spanish], Vidore3IndustrialRetrieval [French], Vidore3IndustrialRetrieval [German], Vidore3IndustrialRetrieval [Italian], Vidore3IndustrialRetrieval [Portuguese], Vidore3IndustrialRetrieval [Spanish], Vidore3PharmaceuticalsRetrieval [French], Vidore3PharmaceuticalsRetrieval [German], Vidore3PharmaceuticalsRetrieval [Italian], Vidore3PharmaceuticalsRetrieval [Portuguese], Vidore3PharmaceuticalsRetrieval [Spanish], Vidore3PhysicsRetrieval [French], Vidore3PhysicsRetrieval [German], Vidore3PhysicsRetrieval [Italian], Vidore3PhysicsRetrieval [Portuguese], Vidore3PhysicsRetrieval [Spanish]

### Overall Scores

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| **Average NDCG@5** | 0.70023 | 0.69611 | -0.59% |
