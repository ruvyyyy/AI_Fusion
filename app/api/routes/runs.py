from fastapi import APIRouter, HTTPException
from app.api.routes.pipeline import run_store   # swap this out after merge
# from app.engines.store import run_store       # uncomment after merge

router = APIRouter()

@router.get("/")
def all_runs():
    return {"runs": list(run_store.values())}

@router.get("/{run_id}/eda")
def get_eda(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    results = run_store[run_id].get("results", {})
    if "eda" not in results:
        raise HTTPException(status_code=404, detail="EDA not ready yet. Check GET /pipeline/{run_id} for progress.")
    return results["eda"]}

@router.get("/{run_id}/training")
def training(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    results = run_store[run_id].get("results", {})
    if "model" not in results:
        raise HTTPException(status_code=404, detail="Training not complete yet. Check GET /pipeline/{run_id} for progress.")
    return results["model"]}

@router.get("/{run_id}/evaluation")
def evaluation(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    results = run_store[run_id].get("results", {})
    if "metrics" not in results:
        raise HTTPException(status_code=404, detail="Evaluation not complete yet. Check GET /pipeline/{run_id} for progress.")
    return results["metrics"]

@router.get("/{run_id}/features")
def features(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    results = run_store[run_id].get("results", {})
    if "importance" not in results:
        raise HTTPException(status_code=404, detail="Feature importance not ready yet. Check GET /pipeline/{run_id} for progress.")
    return results["importance"]

@router.get("/{run_id}/insights")
def insights(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    results = run_store[run_id].get("results", {})
    if not results:
        raise HTTPException(status_code=404, detail="Pipeline not complete yet. Check GET /pipeline/{run_id} for progress.")
    return {
        "run_id": run_id,
        "task_type": run_store[run_id].get("task_type"),
        "status": run_store[run_id].get("status"),
        "progress_pct": run_store[run_id].get("progress_pct"),
        "current_stage": run_store[run_id].get("current_stage"),
        "summary": results.get("eda", {}).get("shape"),
        "metrics": results.get("metrics"),
        "top_features": results.get("importance", [])[:3]
    }


@router.delete("/{run_id}")
def delete(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    del run_store[run_id]
    return {"message": "Run process deleted"}