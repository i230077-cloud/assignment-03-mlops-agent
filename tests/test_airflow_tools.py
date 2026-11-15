"""
Unit & Integration Tests for Apache Airflow Workflow Agent
"""
import pytest
from agent.tools.airflow_tools import AirflowToolkit

@pytest.fixture
def airflow_tool():
    return AirflowToolkit()

def test_list_dags(airflow_tool):
    res = airflow_tool.list_dags()
    assert res.status == "SUCCESS"
    assert res.tool_name == "list_dags"
    assert res.data["total_dags"] >= 1
    dag_ids = [d["dag_id"] for d in res.data["dags"]]
    assert "customer_churn_training_pipeline" in dag_ids

def test_get_dag_run_status(airflow_tool):
    res = airflow_tool.get_dag_run_status("customer_churn_training_pipeline")
    assert res.status == "SUCCESS"
    assert res.data["dag_id"] == "customer_churn_training_pipeline"
    assert res.data["state"] in ["success", "failed", "running"]
    assert "tasks_summary" in res.data

def test_trigger_dag_run_approved(airflow_tool):
    res = airflow_tool.trigger_dag_run("customer_churn_training_pipeline")
    assert res.status == "SUCCESS"
    assert "run_id" in res.data
    assert "validate_dataset" in res.data["stages_scheduled"]
    assert "apply_model_quality_gate" in res.data["stages_scheduled"]

def test_trigger_dag_run_unapproved_blocked(airflow_tool):
    res = airflow_tool.trigger_dag_run("malicious_unapproved_dag")
    assert res.status == "BLOCKED"
    assert "Security Policy" in res.error

def test_get_failed_tasks(airflow_tool):
    res = airflow_tool.get_failed_tasks("customer_churn_training_pipeline", run_id="run_injected_failure")
    assert res.status == "SUCCESS"
    assert res.data["total_failed"] >= 1
    failed_task_ids = [t["task_id"] for t in res.data["failed_tasks"]]
    assert "validate_dataset" in failed_task_ids

def test_get_task_diagnostics(airflow_tool):
    res = airflow_tool.get_task_diagnostics(
        dag_id="customer_churn_training_pipeline",
        run_id="run_injected_failure",
        task_id="validate_dataset",
    )
    assert res.status == "SUCCESS"
    assert res.data["task_id"] == "validate_dataset"
    assert res.data["state"] == "failed"
    assert "suggested_fix" in res.data
    assert len(res.data["suggested_fix"]) > 0
