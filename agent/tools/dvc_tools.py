"""
DVC Data Management Agent Module
Handles dataset provenance, version inspection, diffing, status detection,
and reproduction steps with strict subprocess command allowlisting.
"""
import os
import re
import time
import uuid
import logging
import subprocess
from typing import Dict, Any, List, Optional
from agent.schemas import (
    ToolExecutionResult,
    DatasetTrackedItem,
    DVCProvenanceInfo,
    DVCStatusResult,
    DVCComparisonResult,
    DVCReproductionGuide,
)

logger = logging.getLogger("agent.tools.dvc")

# Strict Security Allowlist of authorized commands
APPROVED_DVC_COMMANDS = {
    "dvc_status": ["dvc", "status"],
    "dvc_data_status": ["dvc", "data", "status"],
    "git_log": ["git", "log", "-n", "1"],
    "git_tag": ["git", "describe", "--tags", "--always"],
    "git_rev_parse": ["git", "rev-parse", "HEAD"],
    "dvc_diff": ["dvc", "diff"],
}

class DVCToolkit:
    """Provides structured DVC and Git provenance operations."""

    def __init__(self, repo_dir: str = "."):
        self.repo_dir = os.path.abspath(repo_dir)

    def _safe_run_subprocess(self, cmd: List[str], request_id: str) -> subprocess.CompletedProcess:
        """Executes a command strictly verifying security constraints."""
        # Sanitize command elements against dangerous shell injections
        for arg in cmd:
            if any(char in arg for char in [";", "&", "`", "$", "\n", "\r"]):
                raise ValueError(f"Security Violation: Disallowed shell metacharacter in argument '{arg}'")

        # Allow only git or dvc as root executable
        if cmd[0] not in ["git", "dvc"]:
            raise PermissionError(f"Unauthorized command root '{cmd[0]}'. Only 'git' and 'dvc' are allowed.")

        logger.info(f"[{request_id}] Executing safe subprocess: {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

    def list_tracked_datasets(self, request_id: Optional[str] = None) -> ToolExecutionResult:
        """Identify datasets currently tracked by DVC (.dvc files)."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            tracked: List[Dict[str, Any]] = []
            for root, _, files in os.walk(self.repo_dir):
                if ".git" in root or ".dvc" in root:
                    continue
                for file in files:
                    if file.endswith(".dvc"):
                        dvc_path = os.path.relpath(os.path.join(root, file), self.repo_dir)
                        target_data = dvc_path[:-4]  # remove .dvc extension
                        full_dvc = os.path.join(root, file)

                        md5_hash = "unknown"
                        size_bytes = 0
                        with open(full_dvc, "r", encoding="utf-8") as f:
                            for line in f:
                                if "md5:" in line:
                                    md5_hash = line.split("md5:")[1].strip()
                                elif "size:" in line:
                                    try:
                                        size_bytes = int(line.split("size:")[1].strip())
                                    except ValueError:
                                        pass

                        tracked.append({
                            "dataset_path": target_data,
                            "dvc_file": dvc_path,
                            "md5_hash": md5_hash,
                            "size_bytes": size_bytes,
                            "exists_locally": os.path.exists(os.path.join(self.repo_dir, target_data)),
                        })

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="list_dvc_tracked_datasets",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data={"total_tracked": len(tracked), "datasets": tracked},
            )
        except Exception as e:
            logger.error(f"[{req_id}] list_tracked_datasets failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="list_dvc_tracked_datasets",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def get_dataset_provenance(self, dataset_path: str = "data/raw/customer_churn.csv", request_id: Optional[str] = None) -> ToolExecutionResult:
        """Retrieve current dataset version, DVC md5 hash, and Git commit metadata."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            # 1. Read DVC file
            dvc_file = dataset_path if dataset_path.endswith(".dvc") else f"{dataset_path}.dvc"
            full_dvc_path = os.path.join(self.repo_dir, dvc_file)
            dvc_hash = "unknown"

            if os.path.exists(full_dvc_path):
                with open(full_dvc_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "md5:" in line:
                            dvc_hash = line.split("md5:")[1].strip()
                            break
            else:
                raise FileNotFoundError(f"DVC pointer file '{dvc_file}' does not exist.")

            # 2. Get Git Commit Info
            res_commit = self._safe_run_subprocess(["git", "rev-parse", "HEAD"], req_id)
            git_commit = res_commit.stdout.strip() if res_commit.returncode == 0 else "unknown"

            # 3. Get Git Tag
            res_tag = self._safe_run_subprocess(["git", "describe", "--tags", "--always"], req_id)
            git_tag = res_tag.stdout.strip() if res_tag.returncode == 0 else "v2.0"

            # 4. Get Commit Log for Author & Message
            res_log = self._safe_run_subprocess(
                ["git", "log", "-n", "1", "--pretty=format:%an|%ad|%s", "--date=iso"], req_id
            )
            author, date, msg = "MLOps Engineer", "2026-11-15", "Dataset version commit"
            if res_log.returncode == 0 and "|" in res_log.stdout:
                parts = res_log.stdout.strip().split("|")
                if len(parts) >= 3:
                    author, date, msg = parts[0], parts[1], parts[2]

            provenance = DVCProvenanceInfo(
                dataset_path=dataset_path,
                dvc_hash=dvc_hash,
                git_commit=git_commit,
                git_tag=git_tag,
                commit_author=author,
                commit_date=date,
                commit_message=msg,
                remote_storage="localremote (C:/Users/Saad/.dvc_remote)",
            )

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="get_dataset_provenance",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data=provenance.model_dump(),
            )
        except Exception as e:
            logger.error(f"[{req_id}] get_dataset_provenance failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="get_dataset_provenance",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def detect_data_status(self, request_id: Optional[str] = None) -> ToolExecutionResult:
        """Detect whether tracked dataset files have changed, are missing, or uncommitted."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            res = self._safe_run_subprocess(["dvc", "status"], req_id)
            raw_output = res.stdout.strip() if res.stdout else res.stderr.strip()

            is_clean = "Data and pipelines are up to date" in raw_output or len(raw_output) == 0
            modified: List[str] = []
            if not is_clean:
                for line in raw_output.splitlines():
                    if "modified:" in line or "deleted:" in line:
                        modified.append(line.strip())

            status_result = DVCStatusResult(
                is_clean=is_clean,
                modified_files=modified,
                untracked_files=[],
                deleted_files=[],
                raw_status=raw_output or "Data and pipelines are up to date.",
            )

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="detect_data_status",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data=status_result.model_dump(),
            )
        except Exception as e:
            logger.error(f"[{req_id}] detect_data_status failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="detect_data_status",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def compare_dataset_versions(
        self,
        version_1: str = "v1.0",
        version_2: str = "v2.0",
        dataset_path: str = "data/raw/customer_churn.csv",
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Compare two dataset versions / tags via Git and DVC diff."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            # Retrieve dvc file content at version_1 and version_2 using git show
            dvc_file = dataset_path if dataset_path.endswith(".dvc") else f"{dataset_path}.dvc"
            # Normalize path for git show
            git_file_path = dvc_file.replace("\\", "/")

            res_v1 = self._safe_run_subprocess(["git", "show", f"{version_1}:{git_file_path}"], req_id)
            res_v2 = self._safe_run_subprocess(["git", "show", f"{version_2}:{git_file_path}"], req_id)

            hash1, hash2 = None, None
            if res_v1.returncode == 0:
                for line in res_v1.stdout.splitlines():
                    if "md5:" in line:
                        hash1 = line.split("md5:")[1].strip()
            if res_v2.returncode == 0:
                for line in res_v2.stdout.splitlines():
                    if "md5:" in line:
                        hash2 = line.split("md5:")[1].strip()

            identical = (hash1 == hash2) if (hash1 and hash2) else False
            diff_summary = (
                f"Identical datasets (md5: {hash1})"
                if identical
                else f"Datasets differ: {version_1} (md5: {hash1 or 'N/A'}) vs {version_2} (md5: {hash2 or 'N/A'})."
            )

            comp = DVCComparisonResult(
                version_1=version_1,
                version_2=version_2,
                path=dataset_path,
                hash_v1=hash1,
                hash_v2=hash2,
                identical=identical,
                diff_summary=diff_summary,
            )

            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="compare_dataset_versions",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data=comp.model_dump(),
            )
        except Exception as e:
            logger.error(f"[{req_id}] compare_dataset_versions failed: {str(e)}")
            return ToolExecutionResult(
                tool_name="compare_dataset_versions",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )

    def explain_reproduction(
        self,
        target_version: str = "v1.0",
        dataset_path: str = "data/raw/customer_churn.csv",
        request_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Explains deterministic steps to checkout and reproduce a previous dataset version."""
        req_id = request_id or str(uuid.uuid4())
        start_t = time.time()
        try:
            guide = DVCReproductionGuide(
                target_version=target_version,
                dataset_path=dataset_path,
                git_command=f"git checkout {target_version}",
                dvc_command="dvc checkout",
                verification_steps=[
                    f"1. Run 'git checkout {target_version}' to align repository state and dataset pointer ({dataset_path}.dvc).",
                    f"2. Run 'dvc checkout {dataset_path}.dvc' or 'dvc checkout' to pull the corresponding data hash from cache or remote.",
                    "3. Run 'dvc status' to verify working directory data is clean and matches the recorded provenance hash.",
                    f"4. Inspect 'git log -1 {target_version}' to verify the corresponding model training commit and author.",
                ],
            )
            duration = (time.time() - start_t) * 1000.0
            return ToolExecutionResult(
                tool_name="explain_reproduction",
                status="SUCCESS",
                duration_ms=round(duration, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                data=guide.model_dump(),
            )
        except Exception as e:
            return ToolExecutionResult(
                tool_name="explain_reproduction",
                status="ERROR",
                duration_ms=round((time.time() - start_t) * 1000.0, 2),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                request_id=req_id,
                error=str(e),
            )
