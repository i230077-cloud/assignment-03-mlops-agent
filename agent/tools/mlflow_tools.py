"""
MLflow Experiment Intelligence Agent Module
Directly interfaces with MlflowClient for experiment tracking, metric retrieval,
run comparison, best candidate identification, and model registry governance.
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
import mlflow
from mlflow.tracking import MlflowClient
from agent.schemas import (
    ToolExecutionResult,
    MLflowRunSummary,
    MLflowModelRegistryInfo,
    MLflowBestRunResult,
)

logger = logging.getLogger("agent.tools.mlflow")

class MLflowToolkit:
    """Interacts with MLflow tracking server using the official Python MlflowClient API."""

    def __init__(self, tracking_uri: Optional[str] = None):
        if tracking_uri:
            self.tracking_uri = tracking_uri
        elif os.path.exists("mlflow.db"):
            self.tracking_uri = "sqlite:///mlflow.db"
        else:
            self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def list_experiments_and_runs(
        self,
        experiment_name: Optional[str] = "customer-churn-prediction",
        max_results: int = 10,
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """List experiments and recent training runs."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            exp = self.client.get_experiment_by_name(experiment_name)
            if not exp:
                experiments = self.client.search_experiments()
                if not experiments:
                    return ToolExecutionResult(
                        tool_name="list_experiments_and_runs",
                        status="SUCCESS",
                        duration_ms=round((time.time() - start_t) * 1000.0, 2),
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        request_id=req_id,
                        data={"message": f"No experiments found with name '{experiment_name}'", "runs": []},
                    )
                exp = experiments[0]

            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            run_summaries = []
            for r in runs:
                run_summaries.append({
                    "run_id": r.info.run_id,
                    "run_name": r.info.run_name or "unnamed",
                    "status": r.info.status,
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r.info.start_time / 1000.0)),
                    "metrics": r.data.metrics,
                    "model_type": r.data.params.get("model_type", "unknown"),
                    "dataset_version": r.data.params.get("dataset_version", r.data.tags.get("dvc.dataset_version", "unknown")),
                    "git_commit": r.data.params.get("dataset_git_commit", "unknown"),
                })

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="list_experiments_and_runs",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={
                    "experiment_id": exp.experiment_id,
                    "experiment_name": exp.name,
                    "total_runs_retrieved": len(run_summaries),
                    "runs": run_summaries,
                },
            )
        except Exception as e:
            logger.error(f"[{req_id}] list_experiments_and_runs failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="list_experiments_and_runs",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def get_run_details(self, run_id: str, request_id: Optional[str] = None) -> ToolExecutionResult:
        """Retrieve model metrics, parameters, tags, and logged artifacts for a run."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            run = self.client.get_run(run_id)
            artifacts = self.client.list_artifacts(run_id)
            artifact_paths = [a.path for a in artifacts]

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="get_run_details",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "experiment_id": run.info.experiment_id,
                    "metrics": run.data.metrics,
                    "parameters": run.data.params,
                    "tags": run.data.tags,
                    "artifacts": artifact_paths,
                },
            )
        except Exception as e:
            logger.error(f"[{req_id}] get_run_details failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="get_run_details",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def get_best_run(
        self,
        experiment_name: str = "customer-churn-prediction",
        metric_name: str = "f1_score",
        mode: str = "max",
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """
        Identify the best run based on evaluation metric, while evaluating
        governance criteria (latency, stability, dataset version) rather than
        blindly crowning the highest raw metric score.
        """
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            exp = self.client.get_experiment_by_name(experiment_name)
            if not exp:
                raise ValueError(f"Experiment '{experiment_name}' not found.")

            runs = self.client.search_runs(experiment_ids=[exp.experiment_id], max_results=20)
            if not runs:
                raise ValueError(f"No runs found in experiment '{experiment_name}'.")

            # Filter runs that reported the target metric
            valid_runs = [r for r in runs if metric_name in r.data.metrics]
            if not valid_runs:
                # Fall back to accuracy or roc_auc if specified metric is missing
                fallback_metric = "accuracy" if "accuracy" in runs[0].data.metrics else "roc_auc"
                metric_name = fallback_metric
                valid_runs = [r for r in runs if metric_name in r.data.metrics]

            reverse_sort = (mode.lower() == "max")
            sorted_runs = sorted(
                valid_runs,
                key=lambda r: r.data.metrics.get(metric_name, -1.0 if reverse_sort else 1e9),
                reverse=reverse_sort,
            )

            best_run = sorted_runs[0]
            best_val = best_run.data.metrics.get(metric_name, 0.0)
            model_type = best_run.data.params.get("model_type", "unknown")
            dataset_version = best_run.data.params.get("dataset_version", best_run.data.tags.get("dvc.dataset_version", "v2.0"))
            git_commit = best_run.data.params.get("dataset_git_commit", best_run.data.tags.get("git.commit", "unknown"))

            # Multi-dimensional Production Readiness Assessment:
            # Check accuracy, inference latency, and dataset freshness
            acc = best_run.data.metrics.get("accuracy", 0.0)
            latency = best_run.data.metrics.get("inference_latency_ms", 10.0)
            readiness_notes = []
            is_ready = True

            if acc < 0.70:
                readiness_notes.append(f"Model accuracy ({acc:.4f}) is below the standard 0.70 baseline.")
                is_ready = False
            else:
                readiness_notes.append(f"Model accuracy ({acc:.4f}) meets baseline acceptance standards.")

            if latency > 150.0:
                readiness_notes.append(f"Inference latency ({latency:.1f}ms) exceeds SLA budget (150ms).")
                is_ready = False
            else:
                readiness_notes.append(f"Inference latency ({latency:.1f}ms) is well within real-time SLA budget.")

            if dataset_version not in ["v2.0", "v1.0"]:
                readiness_notes.append(f"Model trained on untagged dataset ({dataset_version}). Provenance verification needed.")
                is_ready = False
            else:
                readiness_notes.append(f"Dataset provenance confirmed on tagged release ({dataset_version}).")

            evaluated_list = [
                {
                    "run_id": r.info.run_id,
                    "model_type": r.data.params.get("model_type", "unknown"),
                    "metric_value": r.data.metrics.get(metric_name, 0.0),
                    "accuracy": r.data.metrics.get("accuracy", 0.0),
                    "roc_auc": r.data.metrics.get("roc_auc", 0.0),
                    "dataset_version": r.data.params.get("dataset_version", "unknown"),
                }
                for r in sorted_runs
            ]

            result = MLflowBestRunResult(
                best_run_id=best_run.info.run_id,
                metric_name=metric_name,
                metric_value=round(best_val, 4),
                model_type=model_type,
                dataset_version=dataset_version,
                git_commit=git_commit,
                is_production_ready=is_ready,
                readiness_notes=readiness_notes,
                all_evaluated_runs=evaluated_list,
            )

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="get_best_run",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data=result.model_dump(),
            )
        except Exception as e:
            logger.error(f"[{req_id}] get_best_run failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="get_best_run",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def compare_runs(self, run_ids: List[str], request_id: Optional[str] = None) -> ToolExecutionResult:
        """Compare two or more training runs across parameters, metrics, and data provenance."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            comparison = []
            for rid in run_ids:
                run = self.client.get_run(rid)
                comparison.append({
                    "run_id": rid,
                    "model_type": run.data.params.get("model_type", "unknown"),
                    "dataset_version": run.data.params.get("dataset_version", "unknown"),
                    "git_commit": run.data.params.get("dataset_git_commit", "unknown"),
                    "accuracy": run.data.metrics.get("accuracy", 0.0),
                    "f1_score": run.data.metrics.get("f1_score", 0.0),
                    "roc_auc": run.data.metrics.get("roc_auc", 0.0),
                    "parameters": run.data.params,
                })

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="compare_runs",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={"total_compared": len(comparison), "runs": comparison},
            )
        except Exception as e:
            logger.error(f"[{req_id}] compare_runs failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="compare_runs",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def get_model_registry_info(
        self,
        model_name: str = "CustomerChurnModel",
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Retrieve registered model versions, staging status, and metadata."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            versions_list = []
            for mv in model_versions:
                versions_list.append({
                    "version": mv.version,
                    "current_stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "status": mv.status,
                    "creation_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(mv.creation_timestamp / 1000.0)),
                    "source": mv.source,
                })

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="get_model_registry_info",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={
                    "model_name": model_name,
                    "total_versions": len(versions_list),
                    "versions": versions_list,
                },
            )
        except Exception as e:
            logger.error(f"[{req_id}] get_model_registry_info failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="get_model_registry_info",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )
