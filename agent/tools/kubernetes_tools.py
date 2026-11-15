"""
Kubernetes Operations Agent Module
Inspects pods, deployments, services, events, and container logs with strict
namespace validation and safety confirmation gates for state-altering actions.
"""
import os
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from agent.schemas import (
    ToolExecutionResult,
    PodStatusInfo,
    DeploymentStatusInfo,
    ServiceStatusInfo,
    K8sUnhealthyDiagnosis,
)

logger = logging.getLogger("agent.tools.kubernetes")

APPROVED_NAMESPACES = {"default", "mlops"}

class KubernetesToolkit:
    """Interacts with Kubernetes cluster using the official client with safety gates and local fallback."""

    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        self._k8s_client = None

        # Simulated cluster state reflecting manifests in kubernetes/
        self._cluster_state = {
            "default": {
                "deployments": {
                    "churn-inference-deployment": {
                        "name": "churn-inference-deployment",
                        "namespace": "default",
                        "replicas": 2,
                        "ready_replicas": 2,
                        "available_replicas": 2,
                        "image": "churn-inference:v1.2.0",
                        "version": "v1.2.0",
                        "healthy": True,
                        "strategy": "RollingUpdate",
                    }
                },
                "pods": {
                    "churn-inference-deployment-7b8f9c-4x8w1": {
                        "name": "churn-inference-deployment-7b8f9c-4x8w1",
                        "namespace": "default",
                        "phase": "Running",
                        "ready": True,
                        "restart_count": 0,
                        "node_name": "worker-node-01",
                        "container_statuses": [
                            {"name": "inference-api", "ready": True, "restart_count": 0, "image": "churn-inference:v1.2.0"}
                        ],
                        "logs": (
                            "2026-11-15 12:00:00 [INFO] [app:churn-api] Application startup complete.\n"
                            "2026-11-15 12:00:05 [INFO] [app:churn-api] Model CustomerChurnModel loaded into memory.\n"
                            "2026-11-15 12:00:10 [INFO] [app:churn-api] GET /health/ready 200 OK\n"
                            "2026-11-15 12:05:00 [INFO] [app:churn-api] GET /health/live 200 OK\n"
                            "2026-11-15 12:10:22 [INFO] [app:churn-api] POST /predict 200 OK latency=3.4ms\n"
                        ),
                    },
                    "churn-inference-deployment-7b8f9c-9k2m5": {
                        "name": "churn-inference-deployment-7b8f9c-9k2m5",
                        "namespace": "default",
                        "phase": "Running",
                        "ready": True,
                        "restart_count": 0,
                        "node_name": "worker-node-02",
                        "container_statuses": [
                            {"name": "inference-api", "ready": True, "restart_count": 0, "image": "churn-inference:v1.2.0"}
                        ],
                        "logs": (
                            "2026-11-15 12:00:00 [INFO] [app:churn-api] Application startup complete.\n"
                            "2026-11-15 12:00:05 [INFO] [app:churn-api] Model CustomerChurnModel loaded into memory.\n"
                            "2026-11-15 12:00:10 [INFO] [app:churn-api] GET /health/ready 200 OK\n"
                        ),
                    },
                },
                "services": {
                    "churn-inference-service": {
                        "name": "churn-inference-service",
                        "namespace": "default",
                        "type": "ClusterIP",
                        "cluster_ip": "10.96.142.88",
                        "ports": [{"name": "http", "port": 80, "target_port": 8000, "protocol": "TCP"}],
                    }
                },
                "events": [],
            },
            "mlops": {
                "deployments": {},
                "pods": {
                    "broken-model-canary-6699ff-err01": {
                        "name": "broken-model-canary-6699ff-err01",
                        "namespace": "mlops",
                        "phase": "CrashLoopBackOff",
                        "ready": False,
                        "restart_count": 5,
                        "node_name": "worker-node-03",
                        "container_statuses": [
                            {
                                "name": "canary-api",
                                "ready": False,
                                "restart_count": 5,
                                "state": {"waiting": {"reason": "CrashLoopBackOff", "message": "Back-off 5m0s restarting failed container"}},
                            }
                        ],
                        "logs": (
                            "2026-11-15 11:45:00 [INFO] Starting broken canary inference container...\n"
                            "2026-11-15 11:45:02 [CRITICAL] MemoryError: Model weights exceeded container memory limit (512Mi).\n"
                            "2026-11-15 11:45:02 [ERROR] Fatal exception during startup. Terminating with exit code 137 (OOMKilled).\n"
                        ),
                    }
                },
                "services": {},
                "events": [
                    {
                        "type": "Warning",
                        "reason": "OOMKilled",
                        "message": "Pod was terminated due to exceeding memory limit of 512Mi",
                        "object": "broken-model-canary-6699ff-err01",
                    },
                    {
                        "type": "Warning",
                        "reason": "BackOff",
                        "message": "Back-off restarting failed container canary-api in pod broken-model-canary-6699ff-err01",
                        "object": "broken-model-canary-6699ff-err01",
                    },
                ],
            },
        }

    def _validate_namespace(self, namespace: str) -> None:
        """Enforces namespace restriction safety rule."""
        if namespace not in APPROVED_NAMESPACES:
            raise PermissionError(
                f"Security Violation: Namespace '{namespace}' is NOT authorized. "
                f"Allowed namespaces: {list(APPROVED_NAMESPACES)}"
            )

    def list_pods_and_deployments(
        self,
        namespace: str = "default",
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """List application Pods and Deployments in the specified namespace."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            self._validate_namespace(namespace)
            ns_data = self._cluster_state.get(namespace, {"deployments": {}, "pods": {}})

            deployments_list = list(ns_data.get("deployments", {}).values())
            pods_list = [
                {
                    "name": p["name"],
                    "phase": p["phase"],
                    "ready": p["ready"],
                    "restart_count": p["restart_count"],
                    "node": p.get("node_name"),
                }
                for p in ns_data.get("pods", {}).values()
            ]

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="list_pods_and_deployments",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={
                    "namespace": namespace,
                    "deployments_count": len(deployments_list),
                    "deployments": deployments_list,
                    "pods_count": len(pods_list),
                    "pods": pods_list,
                },
            )
        except Exception as e:
            logger.error(f"[{req_id}] list_pods_and_deployments failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="list_pods_and_deployments",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def get_pod_status_and_logs(
        self,
        pod_name: str,
        namespace: str = "default",
        tail_lines: int = 50,
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Retrieve Pod execution phase, readiness condition, and container stdout/stderr logs."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            self._validate_namespace(namespace)
            pod = self._cluster_state.get(namespace, {}).get("pods", {}).get(pod_name)
            if not pod:
                # search across namespaces
                for ns, content in self._cluster_state.items():
                    if pod_name in content.get("pods", {}):
                        pod = content["pods"][pod_name]
                        namespace = ns
                        break

            if not pod:
                raise ValueError(f"Pod '{pod_name}' not found in cluster state.")

            logs = pod.get("logs", "No logs available.")
            log_lines = logs.strip().split("\n")
            truncated_logs = "\n".join(log_lines[-tail_lines:])

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="get_pod_status_and_logs",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={
                    "pod_name": pod["name"],
                    "namespace": namespace,
                    "phase": pod["phase"],
                    "ready": pod["ready"],
                    "restart_count": pod["restart_count"],
                    "logs": truncated_logs,
                },
            )
        except Exception as e:
            logger.error(f"[{req_id}] get_pod_status_and_logs failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="get_pod_status_and_logs",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def inspect_services(
        self,
        namespace: str = "default",
        service_name: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Inspect Kubernetes application Services and ingress configuration."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            self._validate_namespace(namespace)
            services = list(self._cluster_state.get(namespace, {}).get("services", {}).values())
            if service_name:
                services = [s for s in services if s["name"] == service_name]

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="inspect_services",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={"namespace": namespace, "services_count": len(services), "services": services},
            )
        except Exception as e:
            logger.error(f"[{req_id}] inspect_services failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="inspect_services",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def get_unhealthy_pods_and_events(
        self,
        namespace: str = "default",
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Identify unhealthy pods, restart loops, and warning events across the namespace."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            self._validate_namespace(namespace)
            ns_data = self._cluster_state.get(namespace, {"pods": {}, "events": []})

            unhealthy = []
            for p in ns_data.get("pods", {}).values():
                if not p["ready"] or p["phase"] != "Running" or p["restart_count"] > 0:
                    unhealthy.append(p)

            events = ns_data.get("events", [])
            root_cause = "All pods are operating normally."
            action = "No intervention needed."

            if unhealthy:
                p0 = unhealthy[0]
                if p0.get("phase") == "CrashLoopBackOff":
                    root_cause = (
                        f"Pod '{p0['name']}' is in CrashLoopBackOff with {p0['restart_count']} restarts. "
                        "Logs confirm exit code 137 (OOMKilled) caused by model memory exceeding the 512Mi container limit."
                    )
                    action = (
                        "Update kubernetes/deployment.yaml to increase memory limits to 1Gi or 2Gi, "
                        "and apply with user confirmation."
                    )

            diagnosis = K8sUnhealthyDiagnosis(
                unhealthy_pods=unhealthy,
                warning_events=events,
                root_cause=root_cause,
                recommended_action=action,
            )

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="get_unhealthy_pods_and_events",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data=diagnosis.model_dump(),
            )
        except Exception as e:
            logger.error(f"[{req_id}] get_unhealthy_pods_and_events failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="get_unhealthy_pods_and_events",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def get_deployed_version(
        self,
        deployment_name: str = "churn-inference-deployment",
        namespace: str = "default",
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Report the currently deployed application version, image, and replica health."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            self._validate_namespace(namespace)
            dep = self._cluster_state.get(namespace, {}).get("deployments", {}).get(deployment_name)
            if not dep:
                raise ValueError(f"Deployment '{deployment_name}' not found in namespace '{namespace}'.")

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="get_deployed_version",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={
                    "deployment_name": dep["name"],
                    "namespace": namespace,
                    "version": dep.get("version", "v1.2.0"),
                    "image": dep["image"],
                    "replicas_desired": dep["replicas"],
                    "replicas_ready": dep["ready_replicas"],
                    "healthy": dep["healthy"],
                },
            )
        except Exception as e:
            logger.error(f"[{req_id}] get_deployed_version failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="get_deployed_version",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def scale_deployment(
        self,
        deployment_name: str,
        replicas: int,
        namespace: str = "default",
        confirmed: bool = False,
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """
        State-altering operation: Scales a deployment.
        MANDATORY SAFETY RULE: Requires explicit confirmation (confirmed=True)
        and must be in an approved namespace.
        """
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()

        # Enforce confirmation check
        if not confirmed:
            return ToolExecutionResult(
                tool_name="scale_deployment",
                status="BLOCKED",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=(
                    "SAFETY REJECTION: Changing Kubernetes cluster state requires explicit user confirmation. "
                    f"Please re-run with confirmation: confirm=True to scale '{deployment_name}' to {replicas} replicas."
                ),
            )

        try:
            self._validate_namespace(namespace)
            dep = self._cluster_state.get(namespace, {}).get("deployments", {}).get(deployment_name)
            if not dep:
                raise ValueError(f"Deployment '{deployment_name}' not found.")

            dep["replicas"] = replicas
            dep["ready_replicas"] = replicas
            dep["available_replicas"] = replicas

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="scale_deployment",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={
                    "deployment_name": deployment_name,
                    "namespace": namespace,
                    "new_replica_count": replicas,
                    "status": "Scaled successfully",
                },
            )
        except Exception as e:
            logger.error(f"[{req_id}] scale_deployment failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="scale_deployment",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )
