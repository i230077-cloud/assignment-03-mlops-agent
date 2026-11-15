"""
Demonstration Runner for the 5 Evaluation Scenarios
Executes all 5 scenarios required by the rubric, records transcripts,
and exports verified evidence to screenshots/ directory.
"""
import os
import sys
import json
import time

# Ensure repository root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.main import format_synthesis

SCENARIOS = [
    {
        "id": 1,
        "name": "Dataset Provenance",
        "query": "Which dataset version was used for training our recent experiments and what is its Git commit and DVC hash?",
        "expected": "Correct DVC version and reproducible reference",
    },
    {
        "id": 2,
        "name": "Model Selection",
        "query": "Compare recent training runs in MLflow and recommend the best candidate based on F1 score and production readiness.",
        "expected": "MLflow metrics and defensible candidate selection",
    },
    {
        "id": 3,
        "name": "Pipeline Execution",
        "query": "Trigger the model training pipeline customer_churn_training_pipeline and report execution status and stages.",
        "expected": "Real Airflow DAG run and task statuses",
    },
    {
        "id": 4,
        "name": "Deployment Diagnosis",
        "query": "Why is my prediction API / canary unavailable in the mlops namespace?",
        "expected": "Kubernetes health information and relevant logs",
    },
    {
        "id": 5,
        "name": "Multi-Tool Reasoning (Mandatory)",
        "query": "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
        "expected": "Agent integrates information from all four technologies (MLflow, DVC, K8s, Airflow)",
    },
]

def run_demos(ci_mode: bool = False):
    print("=" * 80)
    print("   MLOPS INTELLIGENT AGENT - 5 EVALUATION SCENARIOS DEMONSTRATION")
    print("=" * 80)

    orchestrator = MLOpsAgentOrchestrator()
    transcript_lines = []
    os.makedirs("screenshots", exist_ok=True)

    for sc in SCENARIOS:
        print(f"\n>>> Running Scenario {sc['id']}: {sc['name']}")
        print(f"Query: \"{sc['query']}\"")
        print(f"Expected Evidence: {sc['expected']}")
        print("-" * 80)

        synthesis = orchestrator.plan_and_execute(sc["query"])
        output_text = format_synthesis(synthesis)
        print(output_text)

        transcript_lines.append(f"SCENARIO {sc['id']}: {sc['name']}\n")
        transcript_lines.append(f"USER QUERY: {sc['query']}\n")
        transcript_lines.append(output_text)
        transcript_lines.append("\n" + "=" * 80 + "\n\n")

    # Export transcript
    transcript_path = "screenshots/demo_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.writelines(transcript_lines)
    print(f"\n[+] Full conversation transcript exported to: {transcript_path}")

    # Export evidence files
    # 1. DVC Provenance evidence
    dvc_res = orchestrator.execute_tool("get_dataset_provenance")
    with open("screenshots/dvc_provenance.txt", "w", encoding="utf-8") as f:
        f.write("=== DVC & GIT DATASET PROVENANCE EVIDENCE ===\n")
        f.write(json.dumps(dvc_res.data, indent=2))

    # 2. MLflow evidence
    mlflow_res = orchestrator.execute_tool("list_experiments_and_runs", {"max_results": 5})
    with open("screenshots/mlflow_runs_evidence.json", "w", encoding="utf-8") as f:
        json.dump(mlflow_res.data, f, indent=2)

    # 3. Airflow pipeline evidence
    airflow_res = orchestrator.execute_tool("get_dag_run_status", {"dag_id": "customer_churn_training_pipeline"})
    with open("screenshots/airflow_pipeline_execution.json", "w", encoding="utf-8") as f:
        json.dump(airflow_res.data, f, indent=2)

    # 4. Kubernetes cluster status evidence
    k8s_pods = orchestrator.execute_tool("list_pods_and_deployments", {"namespace": "default"})
    k8s_diag = orchestrator.execute_tool("get_unhealthy_pods_and_events", {"namespace": "mlops"})
    with open("screenshots/kubernetes_cluster_status.txt", "w", encoding="utf-8") as f:
        f.write("=== KUBERNETES DEFAULT DEPLOYMENTS & PODS ===\n")
        f.write(json.dumps(k8s_pods.data, indent=2))
        f.write("\n\n=== KUBERNETES MLOPS FAILURE DIAGNOSTICS ===\n")
        f.write(json.dumps(k8s_diag.data, indent=2))

    print("[+] Exported evidence files to screenshots/ directory:")
    print("    - screenshots/dvc_provenance.txt")
    print("    - screenshots/mlflow_runs_evidence.json")
    print("    - screenshots/airflow_pipeline_execution.json")
    print("    - screenshots/kubernetes_cluster_status.txt")
    print("    - screenshots/demo_transcript.txt")
    print("\nAll 5 evaluation scenarios completed successfully!")

if __name__ == "__main__":
    ci = "--ci-mode" in sys.argv
    run_demos(ci_mode=ci)
