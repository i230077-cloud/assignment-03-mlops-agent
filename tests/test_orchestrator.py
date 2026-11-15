"""
Unit & Integration Tests for MLOps Agent Multi-Tool Orchestrator
"""
import pytest
from agent.orchestrator import MLOpsAgentOrchestrator

@pytest.fixture
def agent():
    return MLOpsAgentOrchestrator()

def test_mandatory_multi_tool_integration_scenario(agent):
    """
    Mandatory Scenario:
    'Find the best recent model, identify the dataset version used to train it,
     check the deployment status, and explain whether retraining is necessary.'
    """
    query = (
        "Find the best recent model, identify the dataset version used to train it, "
        "check the deployment status, and explain whether retraining is necessary."
    )
    res = agent.plan_and_execute(query)

    # Must execute all 4 modules
    executed_tool_names = [t.tool_name for t in res.tool_executions]
    assert "get_best_run" in executed_tool_names
    assert "get_dataset_provenance" in executed_tool_names
    assert "get_deployed_version" in executed_tool_names
    assert "get_dag_run_status" in executed_tool_names

    # Check synthesis components
    assert len(res.known_facts) >= 3
    assert isinstance(res.uncertainties, list)
    assert len(res.recommended_actions) >= 2

    # Retraining MUST NOT be initiated automatically
    assert res.retraining_recommended is False
    assert "NOT INITIATED" in res.retraining_justification

def test_dvc_provenance_scenario(agent):
    query = "Which dataset version was used for recent experiments and what is its Git commit?"
    res = agent.plan_and_execute(query)
    assert any(t.tool_name == "get_dataset_provenance" for t in res.tool_executions)
    assert "customer_churn.csv" in res.direct_answer or "v2.0" in res.direct_answer

def test_dataset_comparison_scenario(agent):
    query = "Compare dataset version v1.0 and v2.0."
    res = agent.plan_and_execute(query)
    assert any(t.tool_name == "compare_dataset_versions" for t in res.tool_executions)
    assert "comparison" in res.direct_answer.lower() or "differ" in res.direct_answer.lower()

def test_model_selection_scenario(agent):
    query = "Compare recent training runs in MLflow and recommend the best candidate based on F1 score."
    res = agent.plan_and_execute(query)
    assert any(t.tool_name in ["get_best_run", "list_experiments_and_runs", "compare_runs"] for t in res.tool_executions)
    assert len(res.known_facts) > 0

def test_airflow_pipeline_execution_scenario(agent):
    query = "Trigger the model training pipeline and report whether execution succeeded."
    res = agent.plan_and_execute(query)
    assert any(t.tool_name == "trigger_dag_run" for t in res.tool_executions)
    assert "customer_churn_training_pipeline" in res.direct_answer

def test_kubernetes_failure_diagnosis_scenario(agent):
    query = "Why is my prediction API unavailable?"
    res = agent.plan_and_execute(query)
    assert any(t.tool_name in ["get_unhealthy_pods_and_events", "get_pod_status_and_logs"] for t in res.tool_executions)
    assert "CrashLoopBackOff" in res.direct_answer or "OOMKilled" in res.direct_answer
