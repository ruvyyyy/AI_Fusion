from fastapi import APIRouter, HTTPException
from app.api.routes.upload import file_store

router = APIRouter()

MODEL_CATALOG = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "task_types": ["classification"],
        "description": "Fast, interpretable linear model. Best when you need to understand which features drive predictions.",
        "hyperparams": {
            "C": {"type": "float", "default": 1.0, "description": "Regularization strength. Lower = stronger regularization."},
            "max_iter": {"type": "int", "default": 100, "description": "Max iterations for the solver."}
        }
    },
    "random_forest_classifier": {
        "name": "Random Forest Classifier",
        "task_types": ["classification"],
        "description": "Ensemble of decision trees. Robust, handles non-linear patterns, gives feature importances automatically.",
        "hyperparams": {
            "n_estimators": {"type": "int", "default": 100, "description": "Number of trees in the forest."},
            "max_depth": {"type": "int", "default": None, "description": "Max depth per tree. None means unlimited."}
        }
    },
    "xgboost_classifier": {
        "name": "XGBoost Classifier",
        "task_types": ["classification"],
        "description": "Gradient boosted trees. Industry standard for tabular data. High accuracy with proper tuning.",
        "hyperparams": {
            "n_estimators": {"type": "int", "default": 100, "description": "Number of boosting rounds."},
            "learning_rate": {"type": "float", "default": 0.1, "description": "Step size per round. Lower is slower but more robust."},
            "max_depth": {"type": "int", "default": 6, "description": "Max tree depth."}
        }
    },
    "linear_regression": {
        "name": "Linear Regression",
        "task_types": ["regression"],
        "description": "The baseline regression model. Fast and interpretable. Assumes features relate linearly to the target.",
        "hyperparams": {
            "fit_intercept": {"type": "bool", "default": True, "description": "Whether to calculate the intercept term."}
        }
    },
    "random_forest_regressor": {
        "name": "Random Forest Regressor",
        "task_types": ["regression"],
        "description": "Same as the classifier variant but predicts continuous values instead of categories.",
        "hyperparams": {
            "n_estimators": {"type": "int", "default": 100, "description": "Number of trees."},
            "max_depth": {"type": "int", "default": None, "description": "Max depth per tree."}
        }
    },
    "kmeans": {
        "name": "K-Means Clustering",
        "task_types": ["clustering"],
        "description": "Groups data into K clusters. Fast and simple. Works best when clusters are roughly spherical.",
        "hyperparams": {
            "n_clusters": {"type": "int", "default": 3, "description": "Number of clusters to form."},
            "max_iter": {"type": "int", "default": 300, "description": "Max iterations per run."}
        }
    },
       "xgboost_regressor": {
        "name": "XGBoost Regressor",
        "task_types": ["regression"],
        "description": "Gradient boosted trees for continuous value prediction. Best default for regression tasks.",
        "hyperparams": {
            "n_estimators": {"type": "int", "default": 100, "description": "Number of boosting rounds."},
            "learning_rate": {"type": "float", "default": 0.1, "description": "Step size per round."},
            "max_depth": {"type": "int", "default": 6, "description": "Max tree depth."}
        }
    },
}

RECOMMENDED_DEFAULTS = {
    "classification": "xgboost_classifier",
    "regression": "xgboost_regressor",
    "clustering": "kmeans"
}

@router.get("/")
def list_models(task_type: str = None):
    if task_type:
        filtered = {
            key: val for key, val in MODEL_CATALOG.items()
            if task_type in val["task_types"]
        }
        if not filtered:
            raise HTTPException(status_code=400, detail=f"Unknown task_type '{task_type}'. Use: classification, regression, clustering.")
        return {"task_type": task_type, "models": filtered}
    return {"models": MODEL_CATALOG}

@router.get("/recommend")
def recommend_model(file_id: str):
    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="File not found. Upload a file first.")
    
    entry = file_store[file_id]
    
    if "task_type" not in entry:
        raise HTTPException(status_code=400, detail="Task type not detected yet. Run /detect first.")
    
    task_type = entry["task_type"]
    recommended_key = RECOMMENDED_DEFAULTS[task_type]
    
    alternatives = {
        key: val for key, val in MODEL_CATALOG.items()
        if task_type in val["task_types"] and key != recommended_key
    }
    
    return {
        "task_type": task_type,
        "recommended": {"model_id": recommended_key, **MODEL_CATALOG[recommended_key]},
        "alternatives": alternatives
    }

@router.get("/{model_id}")
def get_model(model_id: str):
    if model_id not in MODEL_CATALOG:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found. Call GET /models/ to see available models.")
    return {"model_id": model_id, **MODEL_CATALOG[model_id]}

#GET / — full catalog, optional filter
#GET /recommend — best model for a file's task type
#GET /{model_id} — details for one model