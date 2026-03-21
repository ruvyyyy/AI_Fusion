from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import upload, detect, pipeline, runs, models, insights
 
app = FastAPI(title="AI Fusion")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(detect.router, prefix="/detect", tags=["Detect"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
app.include_router(runs.router, prefix="/runs", tags=["Runs"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(insights.router, prefix="/insights", tags=["Insights"])

@app.get("/")
def root():
    return {"message":"AI Fusion is running"}
