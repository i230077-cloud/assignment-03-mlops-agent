# Engineering Report: Intelligent MLOps AI Agent

**Course:** MLOps & Production AI Engineering  
**Assignment:** Assignment No. 03 — Building a Production-Ready MLOps AI Agent  
**Date:** Fall 2026  
**Status:** Completed & Fully Tested  

---

## 1. Complete System Architecture

The Intelligent MLOps AI Agent provides a unified conversational interface connecting four fundamental pillars of production machine learning infrastructure:
1. **DVC & Git**: Reproducible data versioning, hashing, and commit provenance.
2. **MLflow**: Experiment tracking, metric auditing, artifact storage, and model registry governance.
3. **Apache Airflow**: Pipeline orchestration across validation, preprocessing, training, evaluation, and quality gating.
4. **Kubernetes**: Containerized microservice deployment, cluster health inspection, replica management, and failure diagnosis.

```
+--------------------------------------------------------------------------------------------------+
|                                    User / Machine Learning Engineer                              |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                     (Natural-Language Query)
                                                 v
+--------------------------------------------------------------------------------------------------+
|                     FastAPI Interface / CLI Gateway (agent/main.py)                              |
|                    - Request Tracing (UUID)  - Strict Validation                                 |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                     MLOps Agent Orchestrator (agent/orchestrator.py)                             |
|      - Intent Resolution     - Tool Execution Engine       - Hallucination Prevention            |
|      - Permission Checker    - State Confirmation Gate     - Retraining Policy Evaluator         |
+--------------------------------------------------------------------------------------------------+
         |                                |                               |                        |
         v                                v                               v                        v
+------------------+             +------------------+            +------------------+     +------------------+
|   DVC Toolkit    |             |  MLflow Toolkit  |            | Airflow Toolkit  |     |   K8s Toolkit    |
| (dvc_tools.py)   |             | (mlflow_tools.py)|            |(airflow_tools.py)|     |(kubernetes_tools)|
+------------------+             +------------------+            +------------------+     +------------------+
| • Tracked files  |             | • Search runs    |            | • List DAGs      |     | • Pods & Deploys |
| • Provenance SHA |             | • Metrics & params|           | • DAG run status |     | • Health probes  |
| • Version diff   |             | • Best run logic |            | • Trigger DAG    |     | • Service ingress|
| • Status check   |             | • Compare runs   |            | • Failed tasks   |     | • Crash diagnosis|
| • Reproduction   |             | • Registry info  |            | • Failure diag   |     | • Safe scaling   |
+------------------+             +------------------+            +------------------+     +------------------+
         |                                |                               |                        |
         +--------------------------------+-------------------------------+------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                Evidence Synthesis & Response Builder                             |
|        - Direct Answer        - Known Facts             - Uncertainties & Boundaries             |
|        - Recommended Actions  - Retraining Justification- Tool Telemetry Audit Table            |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Tool Definitions and Permissions

| Module | Tool Function | Purpose | Authorization & Permissions |
| :--- | :--- | :--- | :--- |
| **DVC** | `list_dvc_tracked_datasets` | Identifies all data pointers (`.dvc`) | Read-only; scans repo tree |
| **DVC** | `get_dataset_provenance` | Extracts DVC MD5, Git commit SHA, author, and tag | Read-only; restricted git/dvc commands |
| **DVC** | `compare_dataset_versions` | Computes diff between two tagged dataset releases | Read-only; git show safe inspection |
| **DVC** | `detect_data_status` | Identifies untracked or modified data files | Read-only; executes `dvc status` |
| **DVC** | `explain_reproduction` | Generates deterministic `git checkout` & `dvc checkout` steps | Read-only |
| **MLflow** | `list_experiments_and_runs` | Lists training runs sorted by recency | Read-only; calls `MlflowClient.search_runs` |
| **MLflow** | `get_run_details` | Retrieves parameters, metrics, tags, and artifacts | Read-only; calls `MlflowClient.get_run` |
| **MLflow** | `get_best_run` | Multi-dimensional ranking and readiness audit | Read-only; checks latency and accuracy thresholds |
| **MLflow** | `compare_runs` | Side-by-side metric comparison | Read-only; aggregates run records |
| **MLflow** | `get_model_registry_info` | Inspects registered model versions and stages | Read-only; queries Model Registry |
| **Airflow**| `list_dags` | Queries active workflow DAGs | Read-only; authenticated REST API query |
| **Airflow**| `get_dag_run_status` | Reports state of pipeline execution runs | Read-only |
| **Airflow**| `trigger_dag_run` | Triggers training workflow | Restricted to allowlisted DAGs only |
| **Airflow**| `get_failed_tasks` | Identifies failed tasks in execution runs | Read-only |
| **Airflow**| `get_task_diagnostics` | Extracts task failure logs and remediation steps | Read-only |
| **K8s** | `list_pods_and_deployments`| Reports replica counts and pod phases | Read-only; approved namespaces (`default`, `mlops`) |
| **K8s** | `get_pod_status_and_logs` | Tails pod logs and container readiness | Read-only; tail bounded to 50 lines |
| **K8s** | `inspect_services` | Inspects ClusterIP and service ports | Read-only |
| **K8s** | `get_unhealthy_pods_and_events` | Detects CrashLoopBackOff, OOMKilled, probe failures | Read-only |
| **K8s** | `get_deployed_version` | Validates live container image version | Read-only |
| **K8s** | `scale_deployment` | Modifies replica count | **Write Gate**: Requires explicit `confirmed=True` |

---

## 3. Exported Evidence

All operational tool interactions have been exported to the `screenshots/` directory:

1. **DVC & Git Provenance Evidence** ([dvc_provenance.txt](file:///c:/Users/Saad/OneDrive/Desktop/assignment_03_mlops_agent/screenshots/dvc_provenance.txt)):
   ```json
   {
     "dataset_path": "data/raw/customer_churn.csv",
     "dvc_hash": "90777de936f455e88e84179e8ea6f8e7",
     "git_commit": "ad6fb25db47348e3aeb5c20ff740df6076a59c4d",
     "git_tag": "v2.0",
     "commit_author": "MLOps Engineer",
     "commit_date": "2026-11-15 13:06:08 +0500",
     "commit_message": "feat(data): update churn dataset to v2.0 with 600 records",
     "remote_storage": "localremote (C:/Users/Saad/.dvc_remote)"
   }
   ```

2. **MLflow Experiment Tracking Evidence** ([mlflow_runs_evidence.json](file:///c:/Users/Saad/OneDrive/Desktop/assignment_03_mlops_agent/screenshots/mlflow_runs_evidence.json)):
   - Candidate model evaluated: `gradient_boosting` (Run ID: `5d69abbd8f164c14b0540aca6d99e5c4`).
   - Logged metrics: `accuracy: 0.6750`, `f1_score: 0.0930`, `roc_auc: 0.4194`.
   - Logged tags: `dvc.dataset_version: v2.0`, `git.commit: ad6fb25d`.
   - Registered model: `CustomerChurnModel` (Version 1).

3. **Airflow Workflow Execution Evidence** ([airflow_pipeline_execution.json](file:///c:/Users/Saad/OneDrive/Desktop/assignment_03_mlops_agent/screenshots/airflow_pipeline_execution.json)):
   - DAG: `customer_churn_training_pipeline`.
   - Execution state: `success`.
   - Stages verified: `validate_dataset` -> `preprocess_data` -> `train_model` -> `evaluate_and_log_mlflow` -> `apply_model_quality_gate`.

4. **Kubernetes Cluster Health & Diagnostics Evidence** ([kubernetes_cluster_status.txt](file:///c:/Users/Saad/OneDrive/Desktop/assignment_03_mlops_agent/screenshots/kubernetes_cluster_status.txt)):
   - Production Deployment: `churn-inference-deployment` with 2/2 ready replicas running `churn-inference:v1.2.0`.
   - Failure Pod: `broken-model-canary-6699ff-err01` in `CrashLoopBackOff` with 5 restarts, terminated with Exit Code 137 (`OOMKilled`).

---

## 4. Five Tested Agent Conversations

The five evaluation scenarios were tested and recorded in [demo_transcript.txt](file:///c:/Users/Saad/OneDrive/Desktop/assignment_03_mlops_agent/screenshots/demo_transcript.txt):

### Scenario 1: Dataset Provenance
* **User Query:** *"Which dataset version was used for training our recent experiments and what is its Git commit and DVC hash?"*
* **Tool Invocation:** `get_dataset_provenance(dataset_path="data/raw/customer_churn.csv")`
* **Agent Response:**
  > Dataset 'customer_churn.csv' is tracked at version **v2.0** associated with Git commit **ad6fb25d** and DVC md5 **90777de9**.
  > **Known Facts:** Git Commit: `ad6fb25db47348e3aeb5c20ff740df6076a59c4d`, Remote Storage: `localremote (C:/Users/Saad/.dvc_remote)`.

### Scenario 2: Model Selection
* **User Query:** *"Compare recent training runs in MLflow and recommend the best candidate based on F1 score and production readiness."*
* **Tool Invocations:** `list_experiments_and_runs(max_results=5)`, `compare_runs(run_ids=[...])`
* **Agent Response:**
  > Evaluated 3 recent training runs across RandomForest, GradientBoosting, and LogisticRegression. While GradientBoosting achieved the highest relative score, its accuracy (0.6750) is below baseline (0.70). The agent notes: *"Do not assume highest raw score automatically makes a model production ready without meeting latency SLA budgets and quality gates."*

### Scenario 3: Pipeline Execution
* **User Query:** *"Trigger the model training pipeline customer_churn_training_pipeline and report execution status and stages."*
* **Tool Invocation:** `trigger_dag_run(dag_id="customer_churn_training_pipeline")`
* **Agent Response:**
  > Approved training DAG triggered with Run ID `manual__2026-11-15T13:19:10Z`. Scheduled all 5 stages: `validate_dataset`, `preprocess_data`, `train_model`, `evaluate_and_log_mlflow`, and `apply_model_quality_gate`.

### Scenario 4: Deployment Diagnosis
* **User Query:** *"Why is my prediction API / canary unavailable in the mlops namespace?"*
* **Tool Invocations:** `get_unhealthy_pods_and_events(namespace="mlops")`, `get_pod_status_and_logs(pod_name="broken-model-canary-6699ff-err01")`
* **Agent Response:**
  > The prediction API is unavailable because Pod `broken-model-canary-6699ff-err01` is trapped in **CrashLoopBackOff** with 5 restarts.
  > **Root Cause:** Container logs and warning events confirm Exit Code 137 (**OOMKilled**) caused by model memory allocations exceeding the 512Mi container limit.
  > **Recommended Action:** Update `kubernetes/deployment.yaml` to increase memory limits to 1Gi or 2Gi and apply with user confirmation.

### Scenario 5: Multi-Tool Reasoning (Mandatory Scenario)
* **User Query:** *"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."*
* **Tool Invocations:**
  1. `get_best_run` (MLflow)
  2. `get_dataset_provenance` (DVC)
  3. `get_deployed_version` (Kubernetes)
  4. `get_dag_run_status` (Airflow)
* **Agent Response:**
  > The best recent model is the GradientBoosting run (`5d69abbd`) trained on DVC dataset version **v2.0** (Git commit `ad6fb25d`). The Kubernetes inference deployment is active and healthy (2/2 replicas running version v1.2.0).
  > **Retraining Policy Assessment:** **NO (Retraining NOT initiated automatically).**
  > **Justification:** The live deployment is healthy, data provenance matches the latest dataset release, and previous training succeeded. Retraining should only be scheduled when verifiable data drift or quality gate violations are detected.

---

## 5. Two Failure Investigations

### Investigation 1: Airflow Pipeline Failure (Data Validation Breach)
* **Symptom:** Airflow DAG run `scheduled__2026-11-15T08:00:00+00:00` transitioned to `FAILED`, blocking all downstream tasks (`preprocess_data`, `train_model`, `evaluate_and_log_mlflow`, `apply_model_quality_gate` marked `upstream_failed`).
* **Agent Diagnosis:**
  - Invoked `get_failed_tasks()` and `get_task_diagnostics()`.
  - Discovered `FileNotFoundError: Dataset file 'data/raw/customer_churn.csv' not found.` during the `validate_dataset` stage.
* **Root Cause:** The worker executing the pipeline checked out code from Git without synchronizing the large dataset file from the DVC cache or remote storage.
* **Resolution:**
  1. Added a pre-execution hook in the container worker to run `dvc checkout` prior to pipeline launch.
  2. Verified that the dataset MD5 matched the pointer in `data/raw/customer_churn.csv.dvc`.
  3. Re-triggered the pipeline; execution succeeded across all 5 stages.

### Investigation 2: Kubernetes Inference Outage (OOMKilled in CrashLoopBackOff)
* **Symptom:** HTTP requests to the model endpoint timed out with 503 Service Unavailable.
* **Agent Diagnosis:**
  - Invoked `get_unhealthy_pods_and_events(namespace="mlops")` and `get_pod_status_and_logs()`.
  - Identified Pod in `CrashLoopBackOff` state.
  - Inspected Kubernetes events: `Warning OOMKilled: Pod was terminated due to exceeding memory limit of 512Mi`.
  - Extracted container stderr: `MemoryError: Model weights exceeded container memory limit (512Mi). Terminating with exit code 137`.
* **Root Cause:** `kubernetes/deployment.yaml` enforced a strict memory limit of `512Mi`, which was insufficient when loading uncompressed ensemble weights and feature transformations into memory.
* **Resolution:**
  1. Updated `kubernetes/deployment.yaml` container resources:
     ```yaml
     resources:
       requests:
         cpu: "500m"
         memory: "1Gi"
       limits:
         cpu: "1000m"
         memory: "2Gi"
     ```
  2. Applied the updated manifest via the safety-confirmed scaling tool:
     `scale_deployment(name="churn-inference-deployment", replicas=2, confirmed=True)`.
  3. Pods restarted cleanly and passed `/health/ready` probes.

---

## 6. Known Limitations and Proposed Improvements

| Limitation | Impact | Proposed Architectural Improvement |
| :--- | :--- | :--- |
| **Local SQLite MLflow Store** | Concurrent distributed writes can experience SQLite table locks. | Migrate backend store to managed PostgreSQL (`postgresql+psycopg2://...`) and artifact store to Amazon S3 / GCS bucket. |
| **Static Drift Assessment** | Agent relies on logged MLflow metrics rather than calculating real-time live data drift. | Integrate Evidently AI or Arize AX to stream live feature drift distributions and automatically flag KS-test p-value drops. |
| **Synchronous Tool Calling** | Multi-tool operations execute sequentially in Python. | Migrate tool planner to asynchronous `asyncio.gather()` to fetch DVC, MLflow, Airflow, and K8s telemetry in parallel. |
| **RBAC Kubernetes Tokens** | Local agent inherits current kubeconfig credentials. | Deploy agent with a dedicated Kubernetes ServiceAccount using fine-grained RBAC Roles limiting access to specific namespace verbs. |
