from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid
import pandas as pd
import io
from PIL import Image

router = APIRouter()
file_store = {}

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    filename = file.filename

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty. Please upload a file with data.")

    if filename.endswith((".csv", ".xlsx", ".tsv")):
        df = pd.read_csv(io.BytesIO(raw_bytes))
        if df.empty:
            raise HTTPException(status_code=400, detail="File has no data rows. Please upload a file with at least one row.")
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="File needs at least 2 columns to be useful for ML.")

    file_id = str(uuid.uuid4())
    file_store[file_id] = {"filename": filename, "raw_bytes": raw_bytes}
    return {"file_id": file_id, "filename": filename}
    
@router.get("/{file_id}/preview")
def preview_file(file_id: str):
    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="File not found")
    entry = file_store[file_id]
    filename = entry["filename"]
    raw_bytes = entry["raw_bytes"]
    if filename.endswith((".csv", ".xlsx", ".tsv")):
        df = pd.read_csv(io.BytesIO(raw_bytes)) #io.BytesIO(raw_bytes) converts the raw bytes into a file-like object that pandas can read — because pandas expects a file, not raw bytes.
        return {"preview": df.head(20).to_dict()} #we use pandas to read the bytes into a DataFrame, then return the first 20 rows as a dict
    elif filename.endswith((".txt", ".json")):
        text = raw_bytes.decode("utf-8")
        return {"preview": text[:500]} #we decode the bytes into a string and return the first 500 characters
    elif filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
        img = Image.open(io.BytesIO(raw_bytes))
        return {"preview": {"width": img.width, "height": img.height, "format": img.format}}

    else:
        return {"preview": "preview not supported for this file type"}
    
@router.delete("/{file_id}/delete")
def existence(file_id: str):
    if file_id not in file_store:
        raise HTTPException(status_code=404, detail="File not found")
    del file_store[file_id]
    return {"message": "File removed"}

@router.get("/")
def listoffiles():
    return [{"file_id": fid, "filename": entry["filename"]} for fid, entry in file_store.items()] #This is a list comprehension, it loops through the store and builds a clean list.