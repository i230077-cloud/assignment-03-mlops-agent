"""
Unit & Integration Tests for DVC Data Management Agent
"""
import pytest
from agent.tools.dvc_tools import DVCToolkit

@pytest.fixture
def dvc_tool():
    return DVCToolkit()

def test_list_dvc_tracked_datasets(dvc_tool):
    res = dvc_tool.list_tracked_datasets()
    assert res.status == "SUCCESS"
    assert res.tool_name == "list_dvc_tracked_datasets"
    assert "datasets" in res.data
    assert res.data["total_tracked"] >= 1
    # Check customer churn dataset is in list
    paths = [d["dataset_path"] for d in res.data["datasets"]]
    assert any("customer_churn.csv" in p for p in paths)

def test_get_dataset_provenance(dvc_tool):
    res = dvc_tool.get_dataset_provenance("data/raw/customer_churn.csv")
    assert res.status == "SUCCESS"
    assert res.tool_name == "get_dataset_provenance"
    assert res.data["dataset_path"] == "data/raw/customer_churn.csv"
    assert len(res.data["dvc_hash"]) > 0
    assert len(res.data["git_commit"]) > 0
    assert "git_tag" in res.data

def test_compare_dataset_versions(dvc_tool):
    res = dvc_tool.compare_dataset_versions(version_1="v1.0", version_2="v2.0")
    assert res.status == "SUCCESS"
    assert res.tool_name == "compare_dataset_versions"
    assert "diff_summary" in res.data
    assert res.data["version_1"] == "v1.0"
    assert res.data["version_2"] == "v2.0"

def test_detect_data_status(dvc_tool):
    res = dvc_tool.detect_data_status()
    assert res.status == "SUCCESS"
    assert "is_clean" in res.data
    assert "raw_status" in res.data

def test_explain_reproduction(dvc_tool):
    res = dvc_tool.explain_reproduction("v1.0")
    assert res.status == "SUCCESS"
    assert "git checkout v1.0" in res.data["git_command"]
    assert "dvc checkout" in res.data["dvc_command"]
    assert len(res.data["verification_steps"]) >= 3

def test_security_disallowed_command_root(dvc_tool):
    with pytest.raises(PermissionError):
        dvc_tool._safe_run_subprocess(["bash", "-c", "echo hacked"], request_id="sec_test")

def test_security_disallowed_shell_metacharacters(dvc_tool):
    with pytest.raises(ValueError):
        dvc_tool._safe_run_subprocess(["git", "status; rm -rf /"], request_id="sec_test_meta")
