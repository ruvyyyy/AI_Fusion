# from fastapi import APIRouter, HTTPException
# import io
# from app.api.routes.pipeline import run_store

# router = APIRouter()

# @router.get("/")
# def all_runs():
#     return {"runs": list(run_store.values())}

# @router.get("/{run_id}/eda")
# def get_eda(run_id: str):
#     if run_id not in run_store:
#         raise HTTPException(status_code=404, detail="File not found")
#     return {"run_id": run_id, "eda": "distributions, correlations, missing values coming soon"}

# @router.get("/{run_id}/training")
# def training(run_id: str):
#     if run_id not in run_store:
#         raise HTTPException(status_code=404, detail="File not found")
#     return {"run_id": run_id, "training": "selected model, hyperparams, CV scores coming soon"}

# @router.get("/{run_id}/evaluation")
# def evaluation(run_id: str):
#     if run_id not in run_store:
#         raise HTTPException(status_code=404, detail="File not found")
#     return {"run_id": run_id, "evaluation": "R2, MAE, accuracy, F1, confusion matrix coming soon"}

# @router.get("/{run_id}/features")
# def features(run_id: str):
#     if run_id not in run_store:
#         raise HTTPException(status_code=404, detail="File not found")
#     return {"run_id": run_id, "features": "SHAP values, feature importances coming soon"}

# @router.get("/{run_id}/insights")
# def insights(run_id: str):
#     if run_id not in run_store:
#         raise HTTPException(status_code=404, detail="File not found")
#     return {"run_id": run_id, "insights": "LLM narrative insights coming soon"}


# @router.delete("/{run_id}")
# def delete(run_id: str):
#     if run_id not in run_store:
#         raise HTTPException(status_code=404, detail="File not found")
#     del run_store[run_id]
#     return {"message": "Run process deleted"}

#New version with shared helper functions and direct store import to avoid circular imports showing real results instead of placeholders. 
#Earlier runs.py had placeholder responses and imported run_store via pipeline.py which caused circular imports. Now it imports run_store directly from store.py and has helper functions to check run status and results, returning real data stored by executor.py as the pipeline runs.
from fastapi import APIRouter, HTTPException

# from app.api.routes.pipeline import run_store   # swap this out after merge
from app.engines.store import run_store       # uncomment after merge
=======
from app.engines.store import run_store          # ← direct import, not via pipeline.py


router = APIRouter()


def get_run_or_404(run_id: str):
    """ Shared helper — raises 404 if run_id doesn't exist """
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_store[run_id]


def get_result_or_404(run_id: str, key: str):
    """ Shared helper — raises 404 if run not done or result key missing """
    run = get_run_or_404(run_id)

    if run["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Run is not completed yet. Current status: {run['status']}"
        )

    result = run.get("results", {}).get(key)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No '{key}' result found for this run."
        )
    return result


@router.get("/")
def all_runs():
    return {"runs": list(run_store.values())}


@router.get("/{run_id}/eda")
def get_eda(run_id: str):
<<<<<<< HEAD
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
=======
    data = get_result_or_404(run_id, "eda")
    return {"run_id": run_id, "eda": data}


@router.get("/{run_id}/training")
def training(run_id: str):
    data = get_result_or_404(run_id, "model")       # stored as "model" by executor
    return {"run_id": run_id, "training": data}


@router.get("/{run_id}/evaluation")
def evaluation(run_id: str):
    data = get_result_or_404(run_id, "metrics")     # stored as "metrics" by executor
    return {"run_id": run_id, "evaluation": data}


@router.get("/{run_id}/features")
def features(run_id: str):
    data = get_result_or_404(run_id, "importance")  # stored as "importance" by executor
    return {"run_id": run_id, "features": data}


@router.get("/{run_id}/insights")
def insights(run_id: str):
    # insights.py will populate this — for now return what's stored
    run = get_run_or_404(run_id)
    data = run.get("results", {}).get("insights")
    if data is None:
        return {"run_id": run_id, "insights": "Not generated yet. Call POST /insights/{run_id}/generate"}
    return {"run_id": run_id, "insights": data}
>>>>>>> df2699751755e99a8f53503ef6ab6e273cab8f5b


@router.delete("/{run_id}")
def delete(run_id: str):
    get_run_or_404(run_id)
    del run_store[run_id]
    return {"message": "Run deleted"}