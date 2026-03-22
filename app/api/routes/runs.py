from fastapi import APIRouter, HTTPException
import io
from app.api.routes.pipeline import run_store

router = APIRouter()

@router.get("/")
def all_runs():
    return {"runs": list(run_store.values())}

@router.get("/{run_id}/eda")
def get_eda(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    return {"run_id": run_id, "eda": "distributions, correlations, missing values coming soon"}

@router.get("/{run_id}/training")
def training(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    return {"run_id": run_id, "training": "selected model, hyperparams, CV scores coming soon"}

@router.get("/{run_id}/evaluation")
def evaluation(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    return {"run_id": run_id, "evaluation": "R2, MAE, accuracy, F1, confusion matrix coming soon"}

@router.get("/{run_id}/features")
def features(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    return {"run_id": run_id, "features": "SHAP values, feature importances coming soon"}

@router.get("/{run_id}/insights")
def insights(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    return {"run_id": run_id, "insights": "LLM narrative insights coming soon"}


@router.delete("/{run_id}")
def delete(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="File not found")
    del run_store[run_id]
    return {"message": "Run process deleted"}