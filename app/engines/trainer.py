#this willl be our trainer module, which will handle all the ML logic: data preparation, EDA, model training, evaluation, and feature importance extraction. It will be called by the pipeline executor and return structured results for the API to serve.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, f1_score, classification_report
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans

MODEL_MAP = {
    "regression": {
        "linear_regression":        LinearRegression(),
        "random_forest_regressor":  RandomForestRegressor(n_estimators=100, random_state=42),
        "xgboost_regressor":        XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    },
    "classification": {
        "logistic_regression":      LogisticRegression(max_iter=1000, random_state=42),
        "random_forest_classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost_classifier":       XGBClassifier(n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'),
    },
    "clustering": {
        "kmeans": None  # handled separately
    }
}

# 
# app/engines/trainer.py
# This is the core ML engine of the AI Fusion Platform.
# It is responsible for everything that happens between "data is uploaded"
# and "results are ready to show the user."
#
# It does NOT talk to FastAPI, routes, or HTTP — it is a pure Python module.
# The pipeline executor (executor.py) will call functions from here.
#
# Responsibilities:
#   1. prepare_data()       — clean, encode, and split the dataframe
#   2. run_eda()            — compute exploratory statistics before training
#   3. train_model()        — pick and train the correct sklearn/xgboost model
#   4. evaluate_model()     — compute task-appropriate metrics on the test set
#   5. get_feature_importance() — extract which features mattered most
#   6. run_training_pipeline()  — orchestrate all steps and return one result dict
# 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from xgboost import XGBClassifier, XGBRegressor

# 
# MODEL MAP
# Keys must exactly match MODEL_CATALOG keys in app/api/routes/models.py
# so that when the pipeline asks for "xgboost_classifier" we find it here.
# Each value is a fresh model instance with sensible defaults.
# 
MODEL_MAP = {
    "regression": {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "xgboost_regressor": XGBRegressor(
            n_estimators=100, random_state=42, verbosity=0
        ),
    },
    "classification": {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "xgboost_classifier": XGBClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        ),
    },
    "clustering": {
        "kmeans": KMeans(n_clusters=3, random_state=42, n_init=10)
    },
}

# STEP 1 — DATA PREPARATION
# Raw dataframes from user uploads are messy. This function makes them
# safe to train on without the user doing anything manually.
#
# What it handles:
#   - Drops columns where more than 50% of values are missing (useless columns)
#   - Fills remaining numeric nulls with the column median
#   - Fills remaining categorical nulls with the most common value
#   - Label-encodes all non-numeric columns so sklearn can process them
#   - Separates features (X) from the target column (y)
#   - Splits into 80% train / 20% test with a fixed random seed for reproducibility
#
# Returns a dict so the caller always gets all outputs by name, not position.

def prepare_data(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    """
    Cleans the dataframe and splits it into train/test sets.

    Args:
        df         : raw dataframe loaded from the uploaded file
        target_col : the column the model should predict
        task_type  : "regression", "classification", or "clustering"

    Returns:
        dict with keys: X_train, X_test, y_train, y_test, feature_names,
                        label_encoders, dropped_cols, encoding_map
    """

    # Work on a copy so we never mutate the original stored dataframe
    df = df.copy()

    #  Drop high-null columns 
    # A column missing more than 50% of its values adds more noise than signal.
    # We record which columns we dropped so the user can see them in results.
    null_ratios = df.isnull().mean()
    dropped_cols = null_ratios[null_ratios > 0.5].index.tolist()
    df.drop(columns=dropped_cols, inplace=True)

    # If the target column was accidentally dropped (>50% nulls), raise early
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' has too many nulls (>50%) and was removed. "
            "Choose a different target column."
        )

    # Separate target from features 
    # For clustering there is no target column — y will be None
    if task_type == "clustering":
        X = df.copy()
        y = None
    else:
        X = df.drop(columns=[target_col])
        y = df[target_col]

    # Fill remaining nulls 
    # Numeric columns: fill with median (resistant to outliers vs mean)
    # Categorical columns: fill with mode (most frequent value)
    for col in X.columns:
        if X[col].dtype in ["float64", "int64"]:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

    # If y has nulls (regression/classification), drop those rows entirely
    if y is not None and y.isnull().any():
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    #  Encode categorical columns 
    # sklearn models require all-numeric input.
    # LabelEncoder converts ["male","female"] → [0, 1] etc.
    # We store each encoder so results can be decoded back to original labels.
    label_encoders = {}
    encoding_map = {}  # human-readable: {"gender": {"male": 0, "female": 1}}

    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Also encode the target if it is categorical (classification with string labels)
    target_encoder = None
    if y is not None and y.dtype == "object":
        target_encoder = LabelEncoder()
        y = pd.Series(
            target_encoder.fit_transform(y.astype(str)), index=y.index
        )
        label_encoders["__target__"] = target_encoder

    feature_names = list(X.columns)

    #  Train / Test Split 
    # 80/20 split with stratify for classification to keep class balance equal
    # in both sets. Clustering has no y so we only split X.
    if task_type == "clustering":
        # No split needed for clustering — we train on everything
        return {
            "X_train": X,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "feature_names": feature_names,
            "label_encoders": label_encoders,
            "encoding_map": encoding_map,
            "dropped_cols": dropped_cols,
        }

    stratify = y if task_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoders": label_encoders,
        "encoding_map": encoding_map,
        "dropped_cols": dropped_cols,
    }

# STEP 2 — EXPLORATORY DATA ANALYSIS
# Runs before training to give the user a statistical summary of their data.
# All outputs are plain Python dicts/lists so they can be JSON-serialised
# directly by FastAPI without any extra conversion.
#
# Computes:
#   - Shape (rows, columns)
#   - Per-column null counts and percentages
#   - Per-column data type
#   - Numeric column stats: mean, median, std, min, max, skew
#   - Categorical column stats: unique count, top 5 most frequent values
#   - Target distribution (class counts for classification, histogram for regression)
#   - Top 10 most correlated feature pairs (highlights redundant features)
# 
def run_eda(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    """
    Computes exploratory statistics on the raw dataframe before any cleaning.

    Args:
        df         : raw dataframe
        target_col : column being predicted
        task_type  : determines how we describe the target distribution

    Returns:
        dict of EDA results ready for JSON serialisation
    """

    eda = {}

    #  Basic shape ─
    eda["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    #  Null report ─
    null_counts = df.isnull().sum()
    eda["null_report"] = {
        col: {
            "null_count": int(null_counts[col]),
            "null_pct": round(float(null_counts[col] / len(df) * 100), 2),
        }
        for col in df.columns
    }

    #  Column types 
    eda["column_types"] = {col: str(df[col].dtype) for col in df.columns}

    #  Numeric column stats 
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    eda["numeric_stats"] = {}
    for col in numeric_cols:
        eda["numeric_stats"][col] = {
            "mean":   round(float(df[col].mean()), 4),
            "median": round(float(df[col].median()), 4),
            "std":    round(float(df[col].std()), 4),
            "min":    round(float(df[col].min()), 4),
            "max":    round(float(df[col].max()), 4),
            "skew":   round(float(df[col].skew()), 4),
        }

    #  Categorical column stats 
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    eda["categorical_stats"] = {}
    for col in cat_cols:
        top_values = df[col].value_counts().head(5).to_dict()
        eda["categorical_stats"][col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in top_values.items()},
        }

    #  Target distribution ─
    if target_col in df.columns:
        if task_type == "classification":
            # Show how many rows belong to each class
            dist = df[target_col].value_counts().to_dict()
            eda["target_distribution"] = {
                str(k): int(v) for k, v in dist.items()
            }
        else:
            # For regression: show a 10-bucket histogram
            counts, bin_edges = np.histogram(df[target_col].dropna(), bins=10)
            eda["target_distribution"] = {
                "histogram_counts": counts.tolist(),
                "histogram_edges": [round(float(e), 4) for e in bin_edges],
            }

    #  Feature correlation (top 10 pairs) ─
    # Only possible when we have numeric columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        # Get upper triangle only to avoid duplicates
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        corr_pairs = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "correlation"})
            .sort_values("correlation", ascending=False)
            .head(10)
        )
        eda["top_correlations"] = [
            {
                "col_a": row["col_a"],
                "col_b": row["col_b"],
                "correlation": round(float(row["correlation"]), 4),
            }
            for _, row in corr_pairs.iterrows()
        ]
    else:
        eda["top_correlations"] = []

    return eda


# 
# STEP 3 — MODEL TRAINING
# Picks the model from MODEL_MAP using task_type + model_id and trains it.
# If no model_id is given, uses the recommended default for that task type.
#
# For clustering: fits on the full dataset (no train/test split needed)
# For others: fits on X_train / y_train only
# 
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    model_id: str = None,
) -> tuple:
    """
    Trains the specified model and returns (trained_model, model_id_used).

    Args:
        X_train   : feature matrix for training
        y_train   : target values for training (None for clustering)
        task_type : "regression", "classification", or "clustering"
        model_id  : key from MODEL_MAP — if None, uses RECOMMENDED_DEFAULTS

    Returns:
        (trained_model, model_id_used)
    """

    # Default model IDs if none specified
    defaults = {
        "regression":     "xgboost_regressor",
        "classification": "xgboost_classifier",
        "clustering":     "kmeans",
    }

    if not model_id:
        model_id = defaults.get(task_type)

    # Validate that this model exists for this task type
    task_models = MODEL_MAP.get(task_type, {})
    if model_id not in task_models:
        available = list(task_models.keys())
        raise ValueError(
            f"Model '{model_id}' not found for task '{task_type}'. "
            f"Available: {available}"
        )

    model = task_models[model_id]

    # Train — clustering uses fit(X) with no y
    if task_type == "clustering":
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)

    return model, model_id


# 
# STEP 4 — MODEL EVALUATION
# Computes task-appropriate metrics on the held-out test set.
# Never evaluates on training data — that would give falsely optimistic numbers.
#
# Regression  → R², MAE, RMSE  (how close are predictions to real values?)
# Classification → accuracy, F1, precision, recall, per-class report
# Clustering  → inertia, silhouette score  (how tight are the clusters?)
# 
def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
    feature_names: list,
) -> dict:
    """
    Evaluates a trained model and returns a metrics dict.

    Args:
        model        : trained sklearn/xgboost model
        X_test       : held-out feature matrix
        y_test       : held-out true labels
        task_type    : determines which metrics to compute
        feature_names: column names (used for clustering since no y_test)

    Returns:
        dict of metric names → values
    """

    metrics = {}

    if task_type == "regression":
        preds = model.predict(X_test)
        metrics["r2_score"]  = round(float(r2_score(y_test, preds)), 4)
        metrics["mae"]       = round(float(mean_absolute_error(y_test, preds)), 4)
        metrics["rmse"]      = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
        # Sample of actual vs predicted for the frontend to display
        sample_size = min(20, len(y_test))
        metrics["predictions_sample"] = {
            "actual":    y_test.iloc[:sample_size].tolist(),
            "predicted": preds[:sample_size].tolist(),
        }

    elif task_type == "classification":
        preds = model.predict(X_test)
        metrics["accuracy"]  = round(float(accuracy_score(y_test, preds)), 4)
        metrics["f1_score"]  = round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4)
        metrics["precision"] = round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4)
        metrics["recall"]    = round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4)

        # Per-class breakdown (useful for imbalanced datasets)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        # Convert keys to strings for JSON safety
        metrics["classification_report"] = {
            str(k): v for k, v in report.items()
        }

    elif task_type == "clustering":
        # For clustering we have no y_test — evaluate internal quality
        # Inertia: sum of squared distances from each point to its cluster centre
        # Lower inertia = tighter clusters (but decreases with more clusters)
        metrics["inertia"] = round(float(model.inertia_), 4)
        metrics["n_clusters"] = int(model.n_clusters)

        # Silhouette score requires sklearn separately (only import if needed)
        try:
            from sklearn.metrics import silhouette_score
            X_all = X_test  # for clustering X_test is actually the full X
            labels = model.labels_
            if len(set(labels)) > 1:
                sil = silhouette_score(X_all, labels, sample_size=min(5000, len(X_all)))
                metrics["silhouette_score"] = round(float(sil), 4)
        except Exception:
            metrics["silhouette_score"] = None

        # Cluster sizes — how many rows in each cluster
        unique, counts = np.unique(model.labels_, return_counts=True)
        metrics["cluster_sizes"] = {
            f"cluster_{int(k)}": int(v) for k, v in zip(unique, counts)
        }

    return metrics


# 
# STEP 5 — FEATURE IMPORTANCE
# After training, this tells the user which input columns had the most
# influence on the model's predictions.
#
# Tree-based models (Random Forest, XGBoost) give importances directly.
# Linear models (Linear/Logistic Regression) use absolute coefficient values.
# Clustering (KMeans) computes distance of each feature from cluster centres.
# 
def get_feature_importance(
    model, feature_names: list, task_type: str
) -> list:
    """
    Extracts and ranks feature importances from a trained model.

    Returns:
        list of dicts sorted descending: [{"feature": str, "importance": float}, ...]
    """

    importances = None

    # Tree-based models expose .feature_importances_ directly
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # Linear models use coefficients — take absolute value since direction
    # doesn't indicate importance magnitude here
    elif hasattr(model, "coef_"):
        coef = model.coef_
        # Logistic regression coef_ is 2D for multi-class — take row mean
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        importances = np.abs(coef)

    # KMeans: use variance of cluster centre values across clusters per feature
    # Higher variance = feature separates clusters more
    elif task_type == "clustering" and hasattr(model, "cluster_centers_"):
        importances = np.var(model.cluster_centers_, axis=0)

    if importances is None:
        return []

    # Normalise to 0–1 range so all model types are comparable
    total = importances.sum()
    if total > 0:
        importances = importances / total

    result = [
        {
            "feature":    str(feature_names[i]),
            "importance": round(float(importances[i]), 6),
        }
        for i in range(len(feature_names))
    ]

    # Sort highest importance first
    return sorted(result, key=lambda x: x["importance"], reverse=True)


# 
# STEP 6 — MASTER ORCHESTRATOR
# This is the function the executor calls. It runs all steps in order,
# handles errors at each step, and returns a single structured result dict
# that the run_store will save and the /runs endpoints will return.
#
# The result dict has top-level keys matching the /runs endpoint structure:
#   result["eda"]        → returned by GET /runs/{id}/eda
#   result["model"]      → returned by GET /runs/{id}/model
#   result["metrics"]    → returned by GET /runs/{id}/metrics
#   result["importance"] → returned by GET /runs/{id}/explain
# 
def run_training_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    model_id: str = None,
) -> dict:
    """
    Full ML pipeline: EDA → prepare → train → evaluate → importance.

    Args:
        df         : raw dataframe from file_store
        target_col : column to predict
        task_type  : "regression", "classification", or "clustering"
        model_id   : optional specific model — defaults to recommended

    Returns:
        dict with keys: eda, model, metrics, importance, status
    """

    result = {}

    #  EDA (runs on raw data before any cleaning) 
    try:
        result["eda"] = run_eda(df, target_col, task_type)
    except Exception as e:
        result["eda"] = {"error": str(e)}

    #  Data Preparation 
    try:
        prepared = prepare_data(df, target_col, task_type)
    except Exception as e:
        # If we can't prepare data, nothing else can run
        result["status"] = "failed"
        result["error"]  = f"Data preparation failed: {str(e)}"
        return result

    X_train      = prepared["X_train"]
    X_test       = prepared["X_test"]
    y_train      = prepared["y_train"]
    y_test       = prepared["y_test"]
    feature_names = prepared["feature_names"]

    #  Model Training 
    try:
        model, model_id_used = train_model(X_train, y_train, task_type, model_id)
        result["model"] = {
            "model_id":       model_id_used,
            "model_class":    type(model).__name__,
            "task_type":      task_type,
            "target_column":  target_col,
            "feature_count":  len(feature_names),
            "training_rows":  len(X_train),
            "test_rows":      len(X_test) if X_test is not None else 0,
            "dropped_columns": prepared["dropped_cols"],
            "encoding_map":   prepared["encoding_map"],
        }
    except Exception as e:
        result["status"] = "failed"
        result["error"]  = f"Model training failed: {str(e)}"
        return result

    #  Evaluation 
    try:
        # For clustering, we pass X_train as both (no test split)
        eval_X = X_test if X_test is not None else X_train
        eval_y = y_test if y_test is not None else None
        result["metrics"] = evaluate_model(
            model, eval_X, eval_y, task_type, feature_names
        )
    except Exception as e:
        result["metrics"] = {"error": str(e)}

    #  Feature Importance 
    try:
        result["importance"] = get_feature_importance(
            model, feature_names, task_type
        )
    except Exception as e:
        result["importance"] = {"error": str(e)}

    result["status"] = "completed"
    return result


