"""
Unit & Integration Tests for MLflow Experiment Intelligence Agent
"""
import pytest
from agent.tools.mlflow_tools import MLflowToolkit

@pytest.fixture
def mlflow_tool():
    return MLflowToolkit()

def test_list_experiments_and_runs(mlflow_tool):
    res = mlflow_tool.list_experiments_and_runs(experiment_name="customer-churn-prediction")
    assert res.status == "SUCCESS"
    assert res.tool_name == "list_experiments_and_runs"
    assert res.data["total_runs_retrieved"] >= 1
    assert "runs" in res.data

def test_get_best_run(mlflow_tool):
    res = mlflow_tool.get_best_run(experiment_name="customer-churn-prediction", metric_name="f1_score")
    assert res.status == "SUCCESS"
    assert res.tool_name == "get_best_run"
    assert "best_run_id" in res.data
    assert "metric_value" in res.data
    assert "is_production_ready" in res.data
    assert "readiness_notes" in res.data
    assert len(res.data["readiness_notes"]) >= 2

def test_get_run_details(mlflow_tool):
    list_res = mlflow_tool.list_experiments_and_runs()
    assert list_res.status == "SUCCESS"
    run_id = list_res.data["runs"][0]["run_id"]

    res = mlflow_tool.get_run_details(run_id)
    assert res.status == "SUCCESS"
    assert res.data["run_id"] == run_id
    assert "metrics" in res.data
    assert "parameters" in res.data
    assert "tags" in res.data

def test_compare_runs(mlflow_tool):
    list_res = mlflow_tool.list_experiments_and_runs()
    runs = list_res.data.get("runs", [])
    if len(runs) >= 2:
        run_ids = [runs[0]["run_id"], runs[1]["run_id"]]
        res = mlflow_tool.compare_runs(run_ids)
        assert res.status == "SUCCESS"
        assert res.data["total_compared"] == 2
        assert len(res.data["runs"]) == 2

def test_get_model_registry_info(mlflow_tool):
    res = mlflow_tool.get_model_registry_info("CustomerChurnModel")
    assert res.status == "SUCCESS"
    assert res.data["model_name"] == "CustomerChurnModel"
    assert "versions" in res.data

def test_nonexistent_experiment_error_handling(mlflow_tool):
    res = mlflow_tool.get_best_run(experiment_name="nonexistent_exp_xyz")
    assert res.status == "ERROR"
    assert "not found" in res.error.lower()
