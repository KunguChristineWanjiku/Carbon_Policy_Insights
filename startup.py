from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s startup %(message)s")
logger = logging.getLogger("startup")


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _required_artifacts() -> list[str]:
    return [
        "data/feature_cols.json",
        "data/target_cols.json",
        "models/scaler.pkl",
        "models/random_forest.pkl",
        "models/xgboost.pkl",
        "models/ensemble.pkl",
    ]


def _missing_artifacts() -> list[str]:
    return [p for p in _required_artifacts() if not Path(p).exists()]


def _run_pipeline() -> None:
    timeout_raw = os.environ.get("PIPELINE_TIMEOUT_SECONDS", "0")
    timeout = int(timeout_raw) if timeout_raw.isdigit() else 0
    cmd = [sys.executable, "pipeline.py"]
    logger.info("Running pipeline bootstrap: %s", " ".join(cmd))
    if timeout > 0:
        subprocess.run(cmd, check=True, timeout=timeout)
    else:
        subprocess.run(cmd, check=True)
    logger.info("Pipeline bootstrap completed successfully.")


def _start_api() -> None:
    port = os.environ.get("PORT", "8000")
    default_cmd = f"gunicorn app:app --bind 0.0.0.0:{port} --workers 1 --timeout 180"
    cmd_str = os.environ.get("APP_START_CMD", default_cmd)
    args = shlex.split(cmd_str, posix=os.name != "nt")
    logger.info("Starting API: %s", cmd_str)
    os.execvp(args[0], args)


def main() -> None:
    run_pipeline_on_startup = _truthy(os.environ.get("RUN_PIPELINE_ON_STARTUP"), default=True)
    force_retrain = _truthy(os.environ.get("FORCE_PIPELINE_RETRAIN"), default=False)
    missing = _missing_artifacts()

    logger.info(
        "Startup configuration: RUN_PIPELINE_ON_STARTUP=%s FORCE_PIPELINE_RETRAIN=%s missing_artifacts=%s",
        run_pipeline_on_startup,
        force_retrain,
        len(missing),
    )
    if missing:
        logger.info("Missing artifacts: %s", missing)

    if run_pipeline_on_startup and (force_retrain or len(missing) > 0):
        _run_pipeline()
    elif run_pipeline_on_startup:
        logger.info("Pipeline run skipped; all required model artifacts already exist.")
    else:
        logger.info("Pipeline run disabled by RUN_PIPELINE_ON_STARTUP.")

    _start_api()


if __name__ == "__main__":
    main()
