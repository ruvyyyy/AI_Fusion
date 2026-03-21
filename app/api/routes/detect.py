from fastapi import APIRouter, HTTPException
import pandas as pd
import io
from app.api.routes.upload import file_store
from typing import Optional

router = APIRouter()

@router.post("/")
def data_profiler(file_id: str):
    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="File not found")
    entry = file_store[file_id]
    filename= entry["filename"]
    raw_bytes= entry["raw_bytes"]
    if not filename.endswith((".csv", ".xlsx", ".tsv")):
        raise HTTPException(status_code=404, detail="Only tabular files supported for detection") #detect only works on tabular data for now
    df = pd.read_csv(io.BytesIO(raw_bytes))
    target_col = df.columns[-1] #grabs the last column of the DataFrame. Convention is that the last column is usually what you're trying to predict.
    n_unique = df[target_col].nunique() #counts how many unique values are in that column. For example [yes, no, yes, no] has 2 unique values.
    dtype = str(df[target_col].dtype) #checks data type.

    if n_unique==2: # almost certainly classification (yes/no, 0/1, true/false)
        task_type = "classification"
        confidence = 0.97
    elif dtype in ("float64", "int64") and n_unique>20: #numeric with many unique values like predicting salary, price, temperature
        task_type = "regression"
        confidence = 0.85
    else: #no clear target pattern
        task_type = "clustering"
        confidence = 0.55
    #The confidence is just how sure we are about that decision — binary columns are very obvious so 0.97, numeric is pretty clear so 0.85, clustering is a guess so 0.55.
    return {"target_column": target_col, "task_type": task_type,"confidence": confidence, "unique_values": n_unique, "dtype": dtype}

@router.patch("/")
def override_detection(file_id: str, task_type: Optional[str] = None, target_column: Optional[str] = None):
    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="File not found")
    entry = file_store[file_id]
    if task_type is not None:
        entry["task_type"] = task_type
    if target_column is not None: 
        entry["target_column"] = target_column
    return {"task_type": entry.get("task_type"), "target_column": entry.get("target_column")}
