# app/engines/executor.py
# The executor is the bridge between the pipeline route and the trainer.
#
# When POST /pipeline/auto is called:
#   1. pipeline.py creates a run_id and calls execute_pipeline() as a
#      FastAPI background task
#   2. execute_pipeline() loads the file, calls trainer step by step,
#      and updates run_store at each stage so /status shows real progress
#   3. When done, run_store holds all results so /runs endpoints return
#      real data
#
# This file has NO FastAPI imports — it only knows about stores and trainer.

import io
import pandas as pd

# Import the master training function from our trainer engine
from app.engines.trainer import run_training_pipeline

# Import both stores directly from their route files
# file_store  : { file_id → { "content": bytes, "filename": str, ... } }
# run_store   : { run_id  → { "status": str, "stages": {}, ... } }
from app.api.routes.upload import file_store
from app.engines.store import run_store


# HELPER — Load dataframe from file_store
# file_store holds raw file bytes. This function reads those bytes back
# into a pandas DataFrame based on the file format that was detected
# at upload time.
def load_dataframe(file_id: str) -> pd.DataFrame:
    """
    Reads the uploaded file bytes from file_store and returns a DataFrame.

    Supports: csv, xlsx, json
    Raises ValueError if file_id is missing or format is unsupported.
    """

    if file_id not in file_store:
        raise ValueError(f"file_id '{file_id}' not found in file_store.")
    # REPLACE WITH — uses "raw_bytes" and "filename" to match your upload.py
    entry    = file_store[file_id]
    raw_bytes = entry["raw_bytes"]      # ← matches your upload.py key exactly
    filename  = entry.get("filename", "")

    buffer = io.BytesIO(raw_bytes)      # ← was "content", now "raw_bytes"

    if filename.endswith(".csv"):
        return pd.read_csv(buffer)

    elif filename.endswith(".xlsx"):
        return pd.read_excel(buffer)

    elif filename.endswith(".json"):
        return pd.read_json(buffer)

    else:
        raise ValueError(
            f"Cannot load '{filename}'. Supported formats: .csv, .xlsx, .json"
        )


# HELPER — Update run_store for a single stage
# Called after each stage completes so the frontend polling
# GET /pipeline/{run_id}/status can show live progress.
#
# Stage order and their progress percentages:
#   loading   →  10%
#   eda       →  30%
#   training  →  70%
#   evaluation → 90%
#   complete  → 100%
def update_stage(run_id: str, stage: str, pct: int, data: dict = None):
    """
    Updates run_store with the current stage name, progress %, and
    optionally stores stage result data.

    Args:
        run_id : the run being updated
        stage  : human-readable stage name shown in /status
        pct    : progress percentage 0–100
        data   : optional result dict to store under this stage name
    """

    run_store[run_id]["current_stage"]  = stage
    run_store[run_id]["progress_pct"]   = pct
    run_store[run_id]["status"]         = "running"

    # Store stage results under their own key so /runs endpoints
    # can return them individually
    if data is not None:
        run_store[run_id]["results"][stage] = data


# MAIN — execute_pipeline()
# This is the function pipeline.py calls as a BackgroundTask.
#
# It runs synchronously inside a background thread — FastAPI's
# BackgroundTasks run in a thread pool so this won't block the server.
#
# Args:
#   run_id     : created by pipeline.py before calling this
#   file_id    : uploaded file to train on
#   task_type  : "regression" | "classification" | "clustering"
#   target_col : column to predict (None for clustering)
#   model_id   : optional specific model — defaults to recommended
def execute_pipeline(
    run_id: str,
    file_id: str,
    task_type: str,
    target_col: str,
    model_id: str = None,
):
    """
    Orchestrates the full pipeline and updates run_store at each stage.
    Called as a background task from POST /pipeline/auto or /pipeline/manual.
    """

    #  Stage 0: Initialise run in store ─
    # This key must already exist (created by pipeline.py before calling us)
    # We add the results sub-dict that each stage will write into
    run_store[run_id]["results"] = {}
    run_store[run_id]["progress_pct"] = 0

    try:

        #  Stage 1: Load file 
        update_stage(run_id, "loading", 10)

        df = load_dataframe(file_id)

        # For clustering, if user passed a target col it's fine —
        # trainer.py ignores it and uses all columns as features
        if task_type == "clustering" and target_col is None:
            target_col = df.columns[0]  # placeholder, won't be used

        #  Stage 2: EDA 
        # We run EDA separately here so the /status shows a distinct stage
        # before the slower training step begins
        update_stage(run_id, "eda", 30)

        #  Stage 3: Training + Evaluation + Importance ─
        # run_training_pipeline() does EDA, prepare, train, evaluate,
        # importance all in one call and returns a structured result dict
        update_stage(run_id, "training", 50)

        result = run_training_pipeline(
            df         = df,
            target_col = target_col,
            task_type  = task_type,
            model_id   = model_id,
        )

        # Check if trainer itself reported a failure
        if result.get("status") == "failed":
            raise RuntimeError(result.get("error", "Training failed."))

        #  Stage 4: Store results stage by stage 
        # Each key maps to a /runs endpoint:
        #   run_store[run_id]["results"]["eda"]        → GET /runs/{id}/eda
        #   run_store[run_id]["results"]["model"]      → GET /runs/{id}/model
        #   run_store[run_id]["results"]["metrics"]    → GET /runs/{id}/metrics
        #   run_store[run_id]["results"]["importance"] → GET /runs/{id}/explain
        update_stage(run_id, "evaluation", 90)

        update_stage(run_id, "eda",        90, result.get("eda"))
        update_stage(run_id, "model",      90, result.get("model"))
        update_stage(run_id, "metrics",    90, result.get("metrics"))
        update_stage(run_id, "importance", 90, result.get("importance"))

        #  Stage 5: Mark complete 
        run_store[run_id]["status"]        = "completed"
        run_store[run_id]["current_stage"] = "completed"
        run_store[run_id]["progress_pct"]  = 100

    except Exception as e:
        # Any unhandled error marks the run as failed with a message
        # The frontend can read this from GET /pipeline/{run_id}/status
        run_store[run_id]["status"]        = "failed"
        run_store[run_id]["current_stage"] = "failed"
        run_store[run_id]["error"]         = str(e)
        run_store[run_id]["progress_pct"]  = 0