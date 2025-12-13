# Model Comparison Report: 8B Original vs Quantized (All Languages)

## Model Information

| Property | Original | Quantized |
|----------|----------|-----------|
| **Model Name** | TomoroAI/tomoro-colqwen3-embed-8b | shubhamg2208/tomoro-ai-colqwen3-embed-8b-autoawq-w4a16 |
| **Parameters** | 8.0B | 8.0B |
| **Memory Usage** | 16.7 GB | 7.9 GB |
| **Release Date** | 2025-11-26 | 2025-12-12 |

## NDCG@5 Performance Comparison

| Benchmark | Original | Quantized | Difference | % Change |
|-----------|----------|-----------|------------|----------|
| Vidore2BioMedicalLecturesRetrieval [English] | 0.67838 | 0.66814 | -0.01024 | -1.51% |
| Vidore2BioMedicalLecturesRetrieval [French] | 0.64321 | 0.62812 | -0.01509 | -2.35% |
| Vidore2BioMedicalLecturesRetrieval [German] | 0.64826 | 0.65091 | +0.00265 | +0.41% |
| Vidore2BioMedicalLecturesRetrieval [Spanish] | 0.64878 | 0.64890 | +0.00012 | +0.02% |
| Vidore2ESGReportsHLRetrieval [English] | 0.75981 | 0.75315 | -0.00666 | -0.88% |
| Vidore2ESGReportsRetrieval [English] | 0.65488 | 0.63820 | -0.01668 | -2.55% |
| Vidore2ESGReportsRetrieval [French] | 0.59901 | 0.61089 | +0.01188 | +1.98% |
| Vidore2ESGReportsRetrieval [German] | 0.58307 | 0.60317 | +0.02010 | +3.45% |
| Vidore2ESGReportsRetrieval [Spanish] | 0.59135 | 0.64206 | +0.05071 | +8.58% |
| Vidore2EconomicsReportsRetrieval [English] | 0.61587 | 0.59014 | -0.02573 | -4.18% |
| Vidore2EconomicsReportsRetrieval [French] | 0.57608 | 0.52960 | -0.04648 | -8.07% |
| Vidore2EconomicsReportsRetrieval [German] | 0.57701 | 0.55906 | -0.01795 | -3.11% |
| Vidore2EconomicsReportsRetrieval [Spanish] | 0.60940 | 0.55938 | -0.05002 | -8.21% |
| Vidore3ComputerScienceRetrieval [English] | 0.74431 | 0.74155 | -0.00276 | -0.37% |
| Vidore3ComputerScienceRetrieval [French] | 0.71210 | 0.71239 | +0.00029 | +0.04% |
| Vidore3ComputerScienceRetrieval [German] | 0.73190 | 0.72050 | -0.01140 | -1.56% |
| Vidore3ComputerScienceRetrieval [Italian] | 0.70937 | 0.72526 | +0.01589 | +2.24% |
| Vidore3ComputerScienceRetrieval [Portuguese] | 0.72496 | 0.73159 | +0.00663 | +0.91% |
| Vidore3ComputerScienceRetrieval [Spanish] | 0.71888 | 0.72124 | +0.00236 | +0.33% |
| Vidore3EnergyRetrieval [English] | 0.64907 | 0.61018 | -0.03889 | -5.99% |
| Vidore3EnergyRetrieval [French] | 0.66292 | 0.65802 | -0.00490 | -0.74% |
| Vidore3EnergyRetrieval [German] | 0.65172 | 0.62375 | -0.02797 | -4.29% |
| Vidore3EnergyRetrieval [Italian] | 0.66164 | 0.64221 | -0.01943 | -2.94% |
| Vidore3EnergyRetrieval [Portuguese] | 0.66518 | 0.63437 | -0.03081 | -4.63% |
| Vidore3EnergyRetrieval [Spanish] | 0.66824 | 0.64061 | -0.02763 | -4.13% |
| Vidore3FinanceEnRetrieval [English] | 0.68226 | 0.67471 | -0.00755 | -1.11% |
| Vidore3FinanceEnRetrieval [French] | 0.61265 | 0.58862 | -0.02403 | -3.92% |
| Vidore3FinanceEnRetrieval [German] | 0.60931 | 0.59157 | -0.01774 | -2.91% |
| Vidore3FinanceEnRetrieval [Italian] | 0.62274 | 0.61055 | -0.01219 | -1.96% |
| Vidore3FinanceEnRetrieval [Portuguese] | 0.61481 | 0.60038 | -0.01443 | -2.35% |
| Vidore3FinanceEnRetrieval [Spanish] | 0.62669 | 0.61558 | -0.01111 | -1.77% |
| Vidore3FinanceFrRetrieval [English] | 0.45463 | 0.42045 | -0.03418 | -7.52% |
| Vidore3FinanceFrRetrieval [French] | 0.46140 | 0.43176 | -0.02964 | -6.42% |
| Vidore3FinanceFrRetrieval [German] | 0.44498 | 0.42005 | -0.02493 | -5.60% |
| Vidore3FinanceFrRetrieval [Italian] | 0.45387 | 0.41925 | -0.03462 | -7.63% |
| Vidore3FinanceFrRetrieval [Portuguese] | 0.45347 | 0.43487 | -0.01860 | -4.10% |
| Vidore3FinanceFrRetrieval [Spanish] | 0.47142 | 0.43897 | -0.03245 | -6.88% |
| Vidore3HrRetrieval [English] | 0.64208 | 0.60840 | -0.03368 | -5.25% |
| Vidore3HrRetrieval [French] | 0.60851 | 0.55382 | -0.05469 | -8.99% |
| Vidore3HrRetrieval [German] | 0.60223 | 0.57421 | -0.02802 | -4.65% |
| Vidore3HrRetrieval [Italian] | 0.61012 | 0.56235 | -0.04777 | -7.83% |
| Vidore3HrRetrieval [Portuguese] | 0.61859 | 0.57461 | -0.04398 | -7.11% |
| Vidore3HrRetrieval [Spanish] | 0.60888 | 0.57509 | -0.03379 | -5.55% |
| Vidore3IndustrialRetrieval [English] | 0.57657 | 0.57577 | -0.00080 | -0.14% |
| Vidore3IndustrialRetrieval [French] | 0.51532 | 0.50532 | -0.01000 | -1.94% |
| Vidore3IndustrialRetrieval [German] | 0.50657 | 0.51859 | +0.01202 | +2.37% |
| Vidore3IndustrialRetrieval [Italian] | 0.51296 | 0.51327 | +0.00031 | +0.06% |
| Vidore3IndustrialRetrieval [Portuguese] | 0.52053 | 0.51813 | -0.00240 | -0.46% |
| Vidore3IndustrialRetrieval [Spanish] | 0.52668 | 0.52458 | -0.00210 | -0.40% |
| Vidore3PharmaceuticalsRetrieval [English] | 0.66648 | 0.66572 | -0.00076 | -0.11% |
| Vidore3PharmaceuticalsRetrieval [French] | 0.64024 | 0.62789 | -0.01235 | -1.93% |
| Vidore3PharmaceuticalsRetrieval [German] | 0.63307 | 0.63286 | -0.00021 | -0.03% |
| Vidore3PharmaceuticalsRetrieval [Italian] | 0.64081 | 0.63486 | -0.00595 | -0.93% |
| Vidore3PharmaceuticalsRetrieval [Portuguese] | 0.63926 | 0.63734 | -0.00192 | -0.30% |
| Vidore3PharmaceuticalsRetrieval [Spanish] | 0.64837 | 0.64129 | -0.00708 | -1.09% |
| Vidore3PhysicsRetrieval [English] | 0.47473 | 0.46322 | -0.01151 | -2.42% |
| Vidore3PhysicsRetrieval [French] | 0.47655 | 0.46877 | -0.00778 | -1.63% |
| Vidore3PhysicsRetrieval [German] | 0.44946 | 0.45686 | +0.00740 | +1.65% |
| Vidore3PhysicsRetrieval [Italian] | 0.47512 | 0.46902 | -0.00610 | -1.28% |
| Vidore3PhysicsRetrieval [Portuguese] | 0.47751 | 0.46578 | -0.01173 | -2.46% |
| Vidore3PhysicsRetrieval [Spanish] | 0.47441 | 0.47455 | +0.00014 | +0.03% |
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
| **Average** | **0.64247** | **0.63063** | **-0.01183** | **-1.84%** |

## Summary

- **Benchmark files (Original):** 22
- **Benchmark files (Quantized):** 22
- **Total entries evaluated:** 71
- **Entries with improvement:** 14
- **Entries with degradation:** 56
- **Unchanged:** 1

### Overall Scores

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| **Average NDCG@5** | 0.64247 | 0.63063 | -1.84% |
