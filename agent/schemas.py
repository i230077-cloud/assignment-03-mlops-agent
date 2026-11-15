"""
Pydantic Schemas for MLOps AI Agent Inputs, Outputs, and Evidence Synthesis
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Base Tool Execution Models
# -------------------------------------------------------------

class ToolExecutionResult(BaseModel):
    tool_name: str
    status: str = Field(..., description="SUCCESS, ERROR, or BLOCKED")
    duration_ms: float
    timestamp: str
    request_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

# -------------------------------------------------------------
# DVC Schemas
# -------------------------------------------------------------

class DatasetTrackedItem(BaseModel):
    path: str
    dvc_file: str
    md5_hash: str
    size_bytes: Optional[int] = None

class DVCProvenanceInfo(BaseModel):
    dataset_path: str
    dvc_hash: str
    git_commit: str
    git_tag: Optional[str] = None
    commit_author: Optional[str] = None
    commit_date: Optional[str] = None
    commit_message: Optional[str] = None
    remote_storage: Optional[str] = None

class DVCStatusResult(BaseModel):
    is_clean: bool
    modified_files: List[str] = Field(default_factory=list)
    untracked_files: List[str] = Field(default_factory=list)
    deleted_files: List[str] = Field(default_factory=list)
    raw_status: str

class DVCComparisonResult(BaseModel):
    version_1: str
    version_2: str
    path: str
    hash_v1: Optional[str] = None
    hash_v2: Optional[str] = None
    identical: bool
    diff_summary: str

class DVCReproductionGuide(BaseModel):
    target_version: str
    dataset_path: str
    git_command: str
    dvc_command: str
    verification_steps: List[str]

# -------------------------------------------------------------
# MLflow Schemas
# -------------------------------------------------------------

class MLflowRunSummary(BaseModel):
    run_id: str
    run_name: str
    experiment_id: str
    status: str
    start_time: str
    metrics: Dict[str, float]
    parameters: Dict[str, str]
    tags: Dict[str, str]

class MLflowModelRegistryInfo(BaseModel):
    model_name: str
    latest_versions: List[Dict[str, Any]]
    creation_timestamp: Optional[str] = None
    last_updated_timestamp: Optional[str] = None
    description: Optional[str] = None

class MLflowBestRunResult(BaseModel):
    best_run_id: str
    metric_name: str
    metric_value: float
    model_type: Optional[str] = None
    dataset_version: Optional[str] = None
    git_commit: Optional[str] = None
    is_production_ready: bool
    readiness_notes: List[str]
    all_evaluated_runs: List[Dict[str, Any]]

# -------------------------------------------------------------
# Airflow Schemas
# -------------------------------------------------------------

class AirflowDAGInfo(BaseModel):
    dag_id: str
    is_paused: bool
    is_active: bool
    description: Optional[str] = None
    timetable_description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class AirflowDAGRunStatus(BaseModel):
    dag_id: str
    run_id: str
    state: str
    execution_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class AirflowTaskDiagnostic(BaseModel):
    dag_id: str
    run_id: str
    task_id: str
    state: str
    try_number: int
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    suggested_fix: Optional[str] = None
    logs_tail: Optional[str] = None

# -------------------------------------------------------------
# Kubernetes Schemas
# -------------------------------------------------------------

class PodStatusInfo(BaseModel):
    name: str
    namespace: str
    phase: str
    ready: bool
    restart_count: int
    node_name: Optional[str] = None
    container_statuses: List[Dict[str, Any]] = Field(default_factory=list)

class DeploymentStatusInfo(BaseModel):
    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    available_replicas: int
    image: str
    version: Optional[str] = None
    healthy: bool

class ServiceStatusInfo(BaseModel):
    name: str
    namespace: str
    type: str
    cluster_ip: str
    ports: List[Dict[str, Any]]

class K8sUnhealthyDiagnosis(BaseModel):
    unhealthy_pods: List[Dict[str, Any]]
    warning_events: List[Dict[str, Any]]
    root_cause: str
    recommended_action: str

# -------------------------------------------------------------
# Multi-Tool Agent Request & Response Synthesis
# -------------------------------------------------------------

class AgentQuery(BaseModel):
    user_prompt: str
    request_id: Optional[str] = None
    allowed_tools: Optional[List[str]] = None

class EvidenceSynthesis(BaseModel):
    direct_answer: str
    known_facts: List[str]
    uncertainties: List[str]
    recommended_actions: List[str]
    retraining_recommended: bool
    retraining_justification: str
    tool_executions: List[ToolExecutionResult]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
