from fastapi import APIRouter, HTTPException
import io
import pandas as pd
from app.api.routes.upload import file_store
import uuid

router = APIRouter()
run_store = {}

@router.post("/")
def auto_pipeline(file_id: str):
    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="File not found")
    entry =  file_store[file_id]
    if "task_type" not in entry:
        raise HTTPException(status_code=404, detail="Run /detect before pipeline")
    run_id = str(uuid.uuid4())
    run_store[run_id] = {"status": "running", "file_id": file_id, "task_type": entry["task_type"]}
    return {"run_id": run_id, "status": "running"}

@router.post("/manual")
def specification(file_id: str, model: str, target_column: str, task_type: str):
    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="File not found")
    entry = file_store[file_id]
    run_id = str(uuid.uuid4())
    run_store[run_id] = {"status": "running", "file_id": file_id,"task_type": task_type, "target_column": target_column, "model": model}
    return {"run_id": run_id, "status": "running"}

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
