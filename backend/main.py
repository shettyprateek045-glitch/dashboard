import io
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from .yolo_service import yolo_engine

app = FastAPI(
    title="YOLOv11 Vision API Server",
    description="High-performance FastAPI inference backend for YOLOv11 Computer Vision Dashboard",
    version="1.0.0"
)

# Enable CORS for local and web frontend dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "title": "YOLOv11 Vision API",
        "status": "online",
        "docs_url": "/docs",
        "model_loaded": yolo_engine.is_loaded,
        "active_model": yolo_engine.model_path
    }

@app.get("/health")
def health_check():
    return {
        "status": "ready",
        "model": yolo_engine.model_path,
        "is_model_loaded": yolo_engine.is_loaded,
        "device": str(yolo_engine.model.device) if yolo_engine.model else "Mock / CPU"
    }

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.40),
    iou: float = Form(0.45)
):
    """
    Accepts an uploaded image, runs YOLOv11 detection, and returns bounding boxes, 
    confidences, classes, and inference latency in JSON.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = yolo_engine.predict(image=image, confidence=confidence, iou=iou)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/model/load")
def switch_model(model_path: str = Form(...)):
    """Dynamically switch YOLOv11 model weights file (e.g. models/best.pt)."""
    try:
        yolo_engine.load_model(model_path)
        return {
            "status": "success",
            "active_model": model_path,
            "is_loaded": yolo_engine.is_loaded
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")

@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for low-latency live camera streaming.
    Receives JPEG binary bytes from client and sends back detection JSON.
    """
    await websocket.accept()
    try:
        while True:
            # Receive binary frame
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            result = yolo_engine.predict(image=image, confidence=0.35, iou=0.45)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close()
