from fastapi import APIRouter, HTTPException, BackgroundTasks
import io
import pandas as pd
from app.api.routes.upload import file_store
import uuid

#adding executor store to track pipeline runs and their statuses
from app.engines.executor import execute_pipeline
#adding external store to avoid circular imports — this is where run_store lives now
from app.engines.store import run_store
router = APIRouter()



# CHANGE 2 — auto_pipeline route
# Added: target_col param, BackgroundTasks param,
#         richer run_store entry, and the actual executor call
# Manual and all other routes below are UNTOUCHED
@router.post("/")
def auto_pipeline(
    file_id: str,
    target_col: str,                        # ← ADDED: executor needs to know what to predict
    background_tasks: BackgroundTasks,      # ← ADDED: lets FastAPI run executor after response
):
    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="File not found")

    entry = file_store[file_id]

    if "task_type" not in entry:
        raise HTTPException(status_code=400, detail="Run /detect before pipeline")

    task_type = entry["task_type"]
    run_id = str(uuid.uuid4())

    # ── CHANGED: run_store entry now has all fields executor expects ──────────
    # executor.py reads and updates these keys as the pipeline progresses
    run_store[run_id] = {
        "run_id":        run_id,
        "file_id":       file_id,
        "task_type":     task_type,
        "target_col":    target_col,
        "status":        "queued",          # ← was "running", now "queued" until executor starts
        "current_stage": "queued",          # ← ADDED: executor updates this per stage
        "progress_pct":  0,                 # ← ADDED: executor updates 0→100
        "results":       {},                # ← ADDED: executor stores eda/metrics/etc here
        "error":         None,              # ← ADDED: executor writes here on failure
    }

    # ── ADDED: this is the line that actually triggers training ───────────────
    # FastAPI runs this AFTER returning the run_id response below
    # so the client never waits for training to finish
    background_tasks.add_task(
        execute_pipeline,
        run_id     = run_id,
        file_id    = file_id,
        task_type  = task_type,
        target_col = target_col,
        model_id   = None,                  # None = auto-select best model
    )

    return {
        "run_id":   run_id,
        "status":   "queued",
        "message":  "Pipeline started. Poll GET /pipeline/{run_id} for progress."
    }
#Updated the placeholder pipeline route to actually create a run entry and trigger the executor, using FastAPI's BackgroundTasks to run it asynchronously. The run_store entry now includes all the fields that executor.py expects to read and update, allowing real-time progress tracking and result storage.
@router.post("/manual")
def specification(
    file_id: str,
    model: str,
    target_column: str,
    task_type: str,
    background_tasks: BackgroundTasks,      # ← ADD
):
    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="File not found")

    run_id = str(uuid.uuid4())

    run_store[run_id] = {
        "run_id":        run_id,
        "file_id":       file_id,
        "task_type":     task_type,
        "target_col":    target_column,
        "status":        "queued",
        "current_stage": "queued",
        "progress_pct":  0,
        "results":       {},
        "error":         None,
    }

    background_tasks.add_task(              # ← ADD
        execute_pipeline,
        run_id     = run_id,
        file_id    = file_id,
        task_type  = task_type,
        target_col = target_column,
        model_id   = model,                 # ← passes the user's chosen model
    )

    return {"run_id": run_id, "status": "queued"}



@router.get("/{run_id}")
def run_progress(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    return run_store[run_id]

@router.post("/{run_id}/cancel")
def cancelled(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    if run_store[run_id]["status"] in ("completed", "cancelled"):
        raise HTTPException(status_code=400, detail="Run already finished")
    run_store[run_id]["status"] = "cancelled"
    return {"run_id": run_id, "status": "cancelled"}

@router.post("/{run_id}/retry")
def retrying(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    if run_store[run_id]["status"] != ("failed"):
        raise HTTPException(status_code=400, detail="can only retry a failed run")
    run_store[run_id]["status"] = "running"
    return {"run_id": run_id, "status": "running"}    
