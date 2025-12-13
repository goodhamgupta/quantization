# Model Comparison Report: 4B Original vs Quantized (All Languages)

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
| Vidore3FinanceEnRetrieval [French] | 0.58705 | 0.57899 | -0.00806 | -1.37% |
| Vidore3FinanceEnRetrieval [German] | 0.59188 | 0.58591 | -0.00597 | -1.01% |
| Vidore3FinanceEnRetrieval [Italian] | 0.61216 | 0.60419 | -0.00797 | -1.30% |
| Vidore3FinanceEnRetrieval [Portuguese] | 0.59784 | 0.60393 | +0.00609 | +1.02% |
| Vidore3FinanceEnRetrieval [Spanish] | 0.62061 | 0.61163 | -0.00898 | -1.45% |
| Vidore3FinanceFrRetrieval [English] | 0.42003 | 0.42656 | +0.00653 | +1.55% |
| Vidore3FinanceFrRetrieval [French] | 0.43333 | 0.44232 | +0.00899 | +2.07% |
| Vidore3FinanceFrRetrieval [German] | 0.42478 | 0.41560 | -0.00918 | -2.16% |
| Vidore3FinanceFrRetrieval [Italian] | 0.41696 | 0.42866 | +0.01170 | +2.81% |
| Vidore3FinanceFrRetrieval [Portuguese] | 0.43508 | 0.43685 | +0.00177 | +0.41% |
| Vidore3FinanceFrRetrieval [Spanish] | 0.44323 | 0.44006 | -0.00317 | -0.72% |
| Vidore3HrRetrieval [English] | 0.60373 | 0.60522 | +0.00149 | +0.25% |
| Vidore3HrRetrieval [French] | 0.55483 | 0.54596 | -0.00887 | -1.60% |
| Vidore3HrRetrieval [German] | 0.56691 | 0.57108 | +0.00417 | +0.74% |
| Vidore3HrRetrieval [Italian] | 0.54890 | 0.54479 | -0.00411 | -0.75% |
| Vidore3HrRetrieval [Portuguese] | 0.57567 | 0.56240 | -0.01327 | -2.31% |
| Vidore3HrRetrieval [Spanish] | 0.57231 | 0.56132 | -0.01099 | -1.92% |
| Vidore3IndustrialRetrieval [English] | 0.57932 | 0.57386 | -0.00546 | -0.94% |
| Vidore3IndustrialRetrieval [French] | 0.49848 | 0.48950 | -0.00898 | -1.80% |
| Vidore3IndustrialRetrieval [German] | 0.51985 | 0.51944 | -0.00041 | -0.08% |
| Vidore3IndustrialRetrieval [Italian] | 0.50466 | 0.50390 | -0.00076 | -0.15% |
| Vidore3IndustrialRetrieval [Portuguese] | 0.51784 | 0.52326 | +0.00542 | +1.05% |
| Vidore3IndustrialRetrieval [Spanish] | 0.52395 | 0.51623 | -0.00772 | -1.47% |
| Vidore3PharmaceuticalsRetrieval [English] | 0.66048 | 0.66057 | +0.00009 | +0.01% |
| Vidore3PharmaceuticalsRetrieval [French] | 0.62456 | 0.62353 | -0.00103 | -0.16% |
| Vidore3PharmaceuticalsRetrieval [German] | 0.63034 | 0.62336 | -0.00698 | -1.11% |
| Vidore3PharmaceuticalsRetrieval [Italian] | 0.63488 | 0.63439 | -0.00049 | -0.08% |
| Vidore3PharmaceuticalsRetrieval [Portuguese] | 0.63929 | 0.63735 | -0.00194 | -0.30% |
| Vidore3PharmaceuticalsRetrieval [Spanish] | 0.64376 | 0.63614 | -0.00762 | -1.18% |
| Vidore3PhysicsRetrieval [English] | 0.46403 | 0.46300 | -0.00103 | -0.22% |
| Vidore3PhysicsRetrieval [French] | 0.46361 | 0.46243 | -0.00118 | -0.25% |
| Vidore3PhysicsRetrieval [German] | 0.45473 | 0.46366 | +0.00893 | +1.96% |
| Vidore3PhysicsRetrieval [Italian] | 0.46990 | 0.45841 | -0.01149 | -2.45% |
| Vidore3PhysicsRetrieval [Portuguese] | 0.46163 | 0.46000 | -0.00163 | -0.35% |
| Vidore3PhysicsRetrieval [Spanish] | 0.46816 | 0.46955 | +0.00139 | +0.30% |
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
| **Average** | **0.63023** | **0.62670** | **-0.00354** | **-0.56%** |

## Summary

- **Benchmark files (Original):** 22
- **Benchmark files (Quantized):** 22
- **Total entries evaluated:** 71
- **Entries with improvement:** 18
- **Entries with degradation:** 52
- **Unchanged:** 1

### Overall Scores

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| **Average NDCG@5** | 0.63023 | 0.62670 | -0.56% |
