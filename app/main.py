import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.model import predict_video, load_model

app = FastAPI(
    title="Video Action Classifier",
    description="Classifies human actions in videos using EfficientNet + BiLSTM (UCF101)",
    version="1.0.0"
)

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pre-load model on startup
@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/")
def root():
    return {"message": "Video Action Classifier API is live 🚀", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    allowed = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Use mp4/avi.")

    # Save uploaded video to temp location
    filename = f"{uuid.uuid4()}.mp4"
    filepath = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_video(filepath)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up temp file
        if os.path.exists(filepath):
            os.remove(filepath)