from fastapi import APIRouter
from fastapi import UploadFile, File
import uuid

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    filename = file.filename
    file_id = str(uuid.uuid4()) #this is used to create a random id for every file uploaded, if 2 people upload data of same name, server wont crash
    return {"file_id": file_id, "filename": filename}