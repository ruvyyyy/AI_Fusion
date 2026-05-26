from fastapi import APIRouter, HTTPException
import pandas as pd
import io
from app.api.routes.upload import file_store
# from app.api.routes.pipeline import run_store
from app.engines.store import run_store # direct import to avoid circular imports 
import ollama
import re 
router = APIRouter()

insights_store = {}

def _load_dataframe(file_id: str) -> pd.DataFrame:
    entry = file_store[file_id]
    return pd.read_csv(io.BytesIO(entry["raw_bytes"]))


def _clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)   # remove **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)         # remove *italic*
    text = re.sub(r'`(.*?)`', r'\1', text)           # remove `code`
    # Convert numbered list items to newline-separated
    text = re.sub(r'\s+(\d+\.)\s+', r'\n\1 ', text)
    return text.strip()
    

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

def _build_prompt(run: dict, eda: dict) -> str:
    task_type = run.get("task_type")
    target_col = run.get("target_column") or run.get("target_col")
    results = run.get("results", {})
    
    model_info = results.get("model", {})
    metrics = results.get("metrics", {})
    importance = results.get("importance", [])

    top_features = importance[:3]
    top_features_text = ", ".join(
        f"{f['feature']} ({round(f['importance'] * 100, 1)}%)"
        for f in top_features
    ) if top_features else "not available"

    if task_type == "classification":
        metrics_text = f"Accuracy: {metrics.get('accuracy')}, F1: {metrics.get('f1_score')}, Precision: {metrics.get('precision')}, Recall: {metrics.get('recall')}"
    elif task_type == "regression":
        metrics_text = f"R2: {metrics.get('r2_score')}, MAE: {metrics.get('mae')}, RMSE: {metrics.get('rmse')}"
    else:
        metrics_text = f"Inertia: {metrics.get('inertia')}, Silhouette: {metrics.get('silhouette_score')}, Clusters: {metrics.get('n_clusters')}"

    prompt = f"""You are a data analyst explaining ML results to a non-technical user. Be concise and practical.

Task: {task_type}
Target column: {target_col}
Model used: {model_info.get('model_id', 'unknown')}
Trained on {model_info.get('training_rows', '?')} rows, tested on {model_info.get('test_rows', '?')} rows.
Performance: {metrics_text}
Top 3 most important features: {top_features_text}
Dataset: {eda.get('shape', {}).get('summary', '')}
Missing values: {eda.get('missing_values', {}).get('summary', '')}

In 4-5 sentences, explain:
1. How well did the model perform?
2. Which features mattered most and why might that make sense?
3. What should the user investigate or improve next?

Write your analysis in 3-4 plain English sentences as a paragraph.

Then add a new section with exactly this header on its own line: "How to improve:"
Below it, give exactly 3 actionable suggestions. Start each one with "- " (dash then space) and nothing else before the text. No markdown, no numbering, no double dashes. Be specific to this dataset and task, not generic advice."""

    return prompt

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
        raise HTTPException(status_code=404, detail="Generate insights first via POST /insights/{run_id}/generate")

    cached   = insights_store[run_id]
    run      = run_store[run_id]
    eda      = cached["eda"]
    results  = run.get("results", {})
    metrics  = results.get("metrics", {})
    importance = results.get("importance", [])

    top_features = ", ".join(
        f"{f['feature']} ({round(f['importance']*100,1)}%)"
        for f in importance[:5]
    ) if importance else "not available"

    context_prompt = f"""You are a data analyst assistant. Answer the user's question about their ML pipeline results.

Dataset context:
- Task type: {run.get('task_type')}
- Target column: {run.get('target_col') or run.get('target_column')}
- Shape: {eda.get('shape', {}).get('summary', '')}
- Missing values: {eda.get('missing_values', {}).get('summary', '')}
- Target distribution: {eda.get('target', {}).get('summary', '')}
- Top features: {top_features}
- Model metrics: {metrics}

User question: {question}

Answer in 2-3 concise sentences. Be specific to their data, not generic."""

    try:
        response = ollama.chat(
            model="gemma4:e2b",
            options={"temperature": 0.4},
            messages=[{"role": "user", "content": context_prompt}]
        )
        answer = _clean_markdown(response["message"]["content"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    return {"question": question, "answer": answer}

@router.post("/{run_id}/narrate")
def narrate_insights(run_id: str):
    if run_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found.")
    
    run = run_store[run_id]
    
    if run.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Pipeline not complete yet. Current status: {run.get('status')}")
    
    if run_id not in insights_store:
        raise HTTPException(status_code=400, detail="Generate EDA insights first via POST /insights/{run_id}/generate")
    
    eda = insights_store[run_id]["eda"]
    prompt = _build_prompt(run, eda)
    
    try:
        response = ollama.chat(
            model="gemma4:e2b",
            options={"temperature": 0.3},
            messages=[
                {"role": "user", "content": prompt}]
            )
        narrative = response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")
    lines = narrative.strip().split("\n")
    narrative_parts = []
    reccommendation_parts = []
    in_improvements = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "how to improve" in line.lower():
            in_improvements = True
            continue
        if in_improvements and line.startswith("- "):
            reccommendation_parts.append(line[2:].strip())
        elif not in_improvements:
            narrative_parts.append(line)
    
    structured = {
        "insights":[
            {
                "text": " ".join(narrative_parts),
                "source_stat": f"model={run.get('results',{}).get('model',{}).get('model_id','?')}",
                "confidence": 0.85
            }
        ],
        "recommendations": [
            {
                "action": rec,
                "rationale":"Based on performace and feature analysis",
                "priority": "high" if i == 0 else "medium" if i == 1 else "low"
            }
            for i, rec in enumerate(reccommendation_parts)
        ]
    }
    insights_store[run_id]["narrative"] = narrative
    insights_store[run_id]["structured"] = structured
    return structured

# Two things to notice:
# temperature: 0.3 — low temperature means more focused, factual responses. Higher temperature means more creative but less reliable. For data analysis explanations we want factual, so 0.3 is right.
# insights_store[run_id]["narrative"] = narrative — we cache the narrative so the frontend can fetch it again without re-running the LLM.