# # app/engines/executor.py
# # The executor is the bridge between the pipeline route and the trainer.
# #
# # When POST /pipeline/auto is called:
# #   1. pipeline.py creates a run_id and calls execute_pipeline() as a
# #      FastAPI background task
# #   2. execute_pipeline() loads the file, calls trainer step by step,
# #      and updates run_store at each stage so /status shows real progress
# #   3. When done, run_store holds all results so /runs endpoints return
# #      real data
# #
# # This file has NO FastAPI imports — it only knows about stores and trainer.

import io
import pandas as pd
import traceback

from app.engines.trainer import run_training_pipeline, make_serializable
from app.api.routes.upload import file_store
from app.engines.store import run_store


def load_dataframe(file_id: str) -> pd.DataFrame:
    if file_id not in file_store:
        raise ValueError(f"file_id '{file_id}' not found in file_store.")

    entry     = file_store[file_id]
    raw_bytes = entry["raw_bytes"]
    filename  = entry.get("filename", "")
    buffer    = io.BytesIO(raw_bytes)

    if filename.endswith(".csv"):
        return pd.read_csv(buffer)
    elif filename.endswith(".xlsx"):
        return pd.read_excel(buffer)
    elif filename.endswith(".json"):
        return pd.read_json(buffer)
    else:
        raise ValueError(f"Unsupported format: '{filename}'")


def update_stage(run_id: str, stage: str, pct: int, data: dict = None):
    run_store[run_id]["current_stage"] = stage
    run_store[run_id]["progress_pct"]  = pct
    run_store[run_id]["status"]        = "running"
    if data is not None:
        run_store[run_id]["results"][stage] = data


def execute_pipeline(
    run_id: str,
    file_id: str,
    task_type: str,
    target_col: str,
    model_id: str = None,
):
    

    # Stage 0 — initialise
    run_store[run_id]["results"]      = {}
    run_store[run_id]["progress_pct"] = 0

    try:
        # Stage 1 — load file
        update_stage(run_id, "loading", 10)
        df = load_dataframe(file_id)

        if task_type == "clustering" and target_col is None:
            target_col = df.columns[0]

        # Stage 2 — EDA marker
        update_stage(run_id, "eda", 30)

        # Stage 3 — train
        update_stage(run_id, "training", 50)
        result = run_training_pipeline(       # ← result defined HERE
            df         = df,
            target_col = target_col,
            task_type  = task_type,
            model_id   = model_id,
        )

        if result.get("status") == "failed":
            raise RuntimeError(result.get("error", "Training failed."))

        result = make_serializable(result)    # ← converted HERE, after result exists

        # Stage 4 — store results
        update_stage(run_id, "evaluation", 90)
        update_stage(run_id, "eda",        90, result.get("eda"))
        update_stage(run_id, "model",      90, result.get("model"))
        update_stage(run_id, "metrics",    90, result.get("metrics"))
        update_stage(run_id, "importance", 90, result.get("importance"))

        # Stage 5 — complete
        run_store[run_id]["status"]        = "completed"
        run_store[run_id]["current_stage"] = "completed"
        run_store[run_id]["progress_pct"]  = 100
        

    except Exception as e:
        print(f"[EXECUTOR] FAILED → {e}")
        traceback.print_exc()
        run_store[run_id]["status"]        = "failed"
        run_store[run_id]["current_stage"] = "failed"
        run_store[run_id]["error"]         = str(e)
        run_store[run_id]["progress_pct"]  = 0

