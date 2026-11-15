"""
MLOps AI Agent CLI & Interactive Runtime
Allows natural-language querying of MLOps infrastructure across DVC, MLflow, Airflow, and Kubernetes.
"""
import sys
import os
import json
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import EvidenceSynthesis

def format_synthesis(synthesis: EvidenceSynthesis) -> str:
    """Formats structured EvidenceSynthesis into clean markdown report."""
    lines = []
    lines.append("=" * 70)
    lines.append("                MLOPS INTELLIGENT AGENT RESPONSE")
    lines.append("=" * 70)
    lines.append(f"\n### DIRECT ANSWER:\n{synthesis.direct_answer}\n")

    lines.append("### 1. KNOWN FACTS:")
    for f in synthesis.known_facts:
        lines.append(f"  * {f}")

    lines.append("\n### 2. UNCERTAINTIES & BOUNDARIES:")
    if synthesis.uncertainties:
        for u in synthesis.uncertainties:
            lines.append(f"  * {u}")
    else:
        lines.append("  * None identified with current operational telemetry.")

    lines.append("\n### 3. RECOMMENDED ACTIONS:")
    for r in synthesis.recommended_actions:
        lines.append(f"  * {r}")

    lines.append(f"\n### 4. RETRAINING POLICY ASSESSMENT:")
    lines.append(f"  * Retraining Recommended: {'YES' if synthesis.retraining_recommended else 'NO'}")
    lines.append(f"  * Justification: {synthesis.retraining_justification}")

    lines.append("\n### 5. OPERATIONAL EVIDENCE (TOOL EXECUTIONS):")
    lines.append(f"{'Tool Name':<30} | {'Status':<8} | {'Duration (ms)':<14} | {'Timestamp'}")
    lines.append("-" * 75)
    for t in synthesis.tool_executions:
        lines.append(f"{t.tool_name:<30} | {t.status:<8} | {t.duration_ms:<14.2f} | {t.timestamp}")

    lines.append("=" * 70)
    return "\n".join(lines)

def run_cli():
    parser = argparse.ArgumentParser(description="MLOps Intelligent AI Agent CLI")
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Natural language query for the MLOps agent",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output response as structured JSON",
    )
    args = parser.parse_args()

    orchestrator = MLOpsAgentOrchestrator()

    if args.query:
        result = orchestrator.plan_and_execute(args.query)
        if args.json:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print(format_synthesis(result))
    else:
        # Default mandatory query demonstration
        default_query = (
            "Find the best recent model, identify the dataset version used to train it, "
            "check the deployment status, and explain whether retraining is necessary."
        )
        print(f"Executing default mandatory integration query:\n> {default_query}\n")
        result = orchestrator.plan_and_execute(default_query)
        print(format_synthesis(result))

if __name__ == "__main__":
    run_cli()
