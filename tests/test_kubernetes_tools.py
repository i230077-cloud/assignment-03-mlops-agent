"""
Unit & Integration Tests for Kubernetes Operations Agent
"""
import pytest
from agent.tools.kubernetes_tools import KubernetesToolkit

@pytest.fixture
def k8s_tool():
    return KubernetesToolkit()

def test_list_pods_and_deployments(k8s_tool):
    res = k8s_tool.list_pods_and_deployments(namespace="default")
    assert res.status == "SUCCESS"
    assert res.data["deployments_count"] >= 1
    assert res.data["pods_count"] >= 2
    dep_names = [d["name"] for d in res.data["deployments"]]
    assert "churn-inference-deployment" in dep_names

def test_get_pod_status_and_logs(k8s_tool):
    res = k8s_tool.get_pod_status_and_logs(
        pod_name="churn-inference-deployment-7b8f9c-4x8w1",
        namespace="default",
    )
    assert res.status == "SUCCESS"
    assert res.data["ready"] is True
    assert res.data["phase"] == "Running"
    assert "Application startup complete" in res.data["logs"]

def test_inspect_services(k8s_tool):
    res = k8s_tool.inspect_services(namespace="default")
    assert res.status == "SUCCESS"
    assert res.data["services_count"] >= 1
    svc_names = [s["name"] for s in res.data["services"]]
    assert "churn-inference-service" in svc_names

def test_get_unhealthy_pods_and_events(k8s_tool):
    res = k8s_tool.get_unhealthy_pods_and_events(namespace="mlops")
    assert res.status == "SUCCESS"
    assert len(res.data["unhealthy_pods"]) >= 1
    assert "OOMKilled" in res.data["root_cause"]
    assert "deployment.yaml" in res.data["recommended_action"]

def test_get_deployed_version(k8s_tool):
    res = k8s_tool.get_deployed_version(deployment_name="churn-inference-deployment", namespace="default")
    assert res.status == "SUCCESS"
    assert res.data["version"] == "v1.2.0"
    assert res.data["replicas_ready"] >= 2

def test_scale_deployment_unconfirmed_rejected(k8s_tool):
    """MANDATORY SAFETY: Changing cluster state without explicit confirmation must be rejected."""
    res = k8s_tool.scale_deployment(
        deployment_name="churn-inference-deployment",
        replicas=4,
        namespace="default",
        confirmed=False,
    )
    assert res.status == "BLOCKED"
    assert "SAFETY REJECTION" in res.error

def test_scale_deployment_confirmed_approved(k8s_tool):
    res = k8s_tool.scale_deployment(
        deployment_name="churn-inference-deployment",
        replicas=3,
        namespace="default",
        confirmed=True,
    )
    assert res.status == "SUCCESS"
    assert res.data["new_replica_count"] == 3

def test_unauthorized_namespace_rejected(k8s_tool):
    """MANDATORY SAFETY: Operations in non-approved namespaces must be rejected."""
    res = k8s_tool.list_pods_and_deployments(namespace="kube-system")
    assert res.status == "ERROR"
    assert "Security Violation: Namespace 'kube-system' is NOT authorized" in res.error
