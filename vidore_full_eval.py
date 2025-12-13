import os
import mteb
import json


# base_dir = "results_w4a16_4B_autoawq_vidore_full_eval_all_languages"
# base_dir = "results_w4a16_8B_autoawq_vidore_full_eval"
base_dir = "results_tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024"

os.makedirs(base_dir, exist_ok=True)

# model_name = "shubhamg2208/tomoro-ai-colqwen3-embed-4b-autoawq-w4a16"
# model_name = "shubhamg2208/tomoro-ai-colqwen3-embed-8b-autoawq-w4a16"
model_name = "shubhamg2208/tomoro-ai-colqwen3-embed-8b-w4a16-autoawq-seqlen-1024"

model = mteb.get_model(model_name, device="cuda")

tasks = [
    "Vidore2BioMedicalLecturesRetrieval",
    "Vidore2ESGReportsHLRetrieval",
    "Vidore2ESGReportsRetrieval",
    "Vidore2EconomicsReportsRetrieval",
    "Vidore3ComputerScienceRetrieval",
    "Vidore3EnergyRetrieval",
    "Vidore3FinanceEnRetrieval",
    "Vidore3FinanceFrRetrieval",
    "Vidore3HrRetrieval",
    "Vidore3IndustrialRetrieval",
    "Vidore3PharmaceuticalsRetrieval",
    "Vidore3PhysicsRetrieval",
    "VidoreArxivQARetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreShiftProjectRetrieval",
    "VidoreSyntheticDocQAAIRetrieval",
    "VidoreSyntheticDocQAEnergyRetrieval",
    "VidoreSyntheticDocQAGovernmentReportsRetrieval",
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
    "VidoreTabfquadRetrieval",
    "VidoreTatdqaRetrieval",
]

for candidate_task in tasks:
    current_task = mteb.get_task(candidate_task)
    print("#" * 100)
    print(f"Evaluating task: {candidate_task}")
    print("#" * 100)
    results = mteb.evaluate(model, current_task, encode_kwargs={"batch_size": 12})
    filename = f"{base_dir}/{candidate_task}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results.task_results[0].to_dict(), f, ensure_ascii=False, indent=2)
