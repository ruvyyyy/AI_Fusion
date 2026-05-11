from fastapi import APIRouter, HTTPException
import pandas as pd
import io
from app.api.routes.upload import file_store
from app.api.routes.pipeline import run_store

router = APIRouter()

insights_store = {}

def _load_dataframe(file_id: str) -> pd.DataFrame:
    entry = file_store[file_id]
    return pd.read_csv(io.BytesIO(entry["raw_bytes"]))

def _generate_eda_insights(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    insights = {}

    # Shape
    insights["shape"] = {
        "summary": f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.",
        "rows": df.shape[0],
        "cols": df.shape[1]
    }

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].to_dict()
    if missing_cols:
        worst = max(missing_cols, key=missing_cols.get)
        insights["missing_values"] = {
            "summary": f"{len(missing_cols)} column(s) have missing values. '{worst}' is the most affected with {missing_cols[worst]} nulls.",
            "columns": {col: {"count": int(v), "pct": round(v / df.shape[0] * 100, 1)} for col, v in missing_cols.items()}
        }
    else:
        insights["missing_values"] = {
            "summary": "No missing values found. The dataset is complete.",
            "columns": {}
        }

    # Target column analysis
    if target_col in df.columns:
        target = df[target_col]
        if task_type == "classification":
            dist = target.value_counts().to_dict()
            total = sum(dist.values())
            pcts = {str(k): round(v / total * 100, 1) for k, v in dist.items()}
            dominant = max(pcts, key=pcts.get)
            balance_note = "balanced" if max(pcts.values()) < 65 else "imbalanced"
            insights["target"] = {
                "summary": f"Target '{target_col}' has {len(dist)} classes and appears {balance_note}. Dominant class: '{dominant}' at {pcts[dominant]}%.",
                "distribution": pcts,
                "balance": balance_note
            }
        elif task_type == "regression":
            insights["target"] = {
                "summary": f"Target '{target_col}' ranges from {round(target.min(), 2)} to {round(target.max(), 2)} with a mean of {round(target.mean(), 2)}.",
                "min": round(float(target.min()), 4),
                "max": round(float(target.max()), 4),
                "mean": round(float(target.mean()), 4),
                "std": round(float(target.std()), 4)
            }
        else:
            insights["target"] = {
                "summary": "Clustering task — no explicit target column to analyze."
            }

    # Numeric feature correlations
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if numeric_cols:
        top_corr = None
        if task_type in ("regression", "classification"):
            try:
                corr = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
                top_corr = {"feature": corr.index[0], "correlation": round(float(corr.iloc[0]), 3)}
            except Exception:
                pass
        insights["numeric_features"] = {
            "summary": f"{len(numeric_cols)} numeric feature(s) found." + (
                f" '{top_corr['feature']}' is most correlated with the target (r={top_corr['correlation']})." if top_corr else ""
            ),
            "count": len(numeric_cols),
            "top_correlation": top_corr
        }

    # Categorical features
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    if cat_cols:
        high_card = [c for c in cat_cols if df[c].nunique() > 20]
        insights["categorical_features"] = {
            "summary": f"{len(cat_cols)} categorical feature(s) found." + (
                f" {len(high_card)} have high cardinality (>20 unique values) and may need encoding." if high_card else " All have manageable cardinality."
            ),
            "count": len(cat_cols),
            "high_cardinality": high_card
        }

    return insights 

@router.post("/{run_id}/generate")
def generate_insights(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found.")
    
    run = run_store[run_id]
    file_id = run.get("file_id")
    task_type = run.get("task_type")

    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="Original file no longer in memory.")

    entry = file_store[file_id]
    if not entry["filename"].endswith((".csv", ".xlsx", ".tsv")):
        raise HTTPException(status_code=400, detail="Insights only available for tabular data.")

    df = _load_dataframe(file_id)

    target_col = (
        run.get("target_column")
        or entry.get("target_column")
        or df.columns[-1]
    )

    eda = _generate_eda_insights(df, target_col, task_type)
    
    result = {
        "run_id": run_id,
        "file_id": file_id,
        "task_type": task_type,
        "target_column": target_col,
        "eda": eda
    }
    insights_store[run_id] = result
    return result

@router.get("/{run_id}")
def get_insights(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found.")
    if run_id not in insights_store:
        raise HTTPException(
            status_code=404,
            detail="Insights not generated yet. Call POST /insights/{run_id}/generate first."
        )
    return insights_store[run_id]

@router.post("/{run_id}/ask")
def ask_question(run_id: str, question: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found.")
    if run_id not in insights_store:
        raise HTTPException(status_code=404, detail="Generate insights first via POST /insights/{run_id}/generate.")

    cached = insights_store[run_id]
    eda = cached["eda"]
    q = question.lower()

    if any(kw in q for kw in ["missing", "null", "empty", "incomplete"]):
        return {"question": question, "answer": eda["missing_values"]["summary"]}

    elif any(kw in q for kw in ["target", "predict", "label", "class"]):
        return {"question": question, "answer": eda.get("target", {}).get("summary", "No target info available.")}

    elif any(kw in q for kw in ["correlat", "important", "feature", "which column"]):
        return {"question": question, "answer": eda.get("numeric_features", {}).get("summary", "No numeric feature analysis available.")}

    elif any(kw in q for kw in ["categor", "text", "string", "encode"]):
        return {"question": question, "answer": eda.get("categorical_features", {}).get("summary", "No categorical feature analysis available.")}

    elif any(kw in q for kw in ["size", "rows", "shape", "how many", "big"]):
        return {"question": question, "answer": eda["shape"]["summary"]}

    elif any(kw in q for kw in ["imbalance", "balance", "skew"]):
        target = eda.get("target", {})
        balance = target.get("balance")
        if balance:
            return {"question": question, "answer": f"The target is {balance}. {target['summary']}"}
        return {"question": question, "answer": "Balance info only available for classification tasks."}

    else:
        return {
            "question": question,
            "answer": "Could not match a specific insight. LLM-powered Q&A slots in here next.",
            "eda": eda
        }