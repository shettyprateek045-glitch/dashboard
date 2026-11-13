import os
import time
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger("yolov11-service")
logging.basicConfig(level=logging.INFO)

class YOLOv11Service:
    def __init__(self, model_path: str = "models/yolo11n.pt", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_loaded = False
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Loads Ultralytics YOLOv11 weights if available, or flags fallback."""
        self.model_path = model_path
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv11 model from: {model_path}")
            self.model = YOLO(model_path)
            self.is_loaded = True
            logger.info(f"YOLOv11 model loaded successfully on device: {self.model.device}")
        except ImportError:
            logger.warning(
                "Ultralytics is not installed. Running in mock fallback mode. "
                "Install with: pip install ultralytics"
            )
            self.model = None
            self.is_loaded = False
        except Exception as e:
            logger.warning(f"Could not load model file {model_path}: {e}. Running in simulated mode.")
            self.model = None
            self.is_loaded = False

    def predict(
        self, 
        image: Image.Image, 
        confidence: float = 0.40, 
        iou: float = 0.45,
        classes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Runs YOLOv11 inference on a PIL image.
        Returns bounding boxes, labels, confidence scores, and timing breakdown.
        """
        start_time = time.perf_counter()

        if self.is_loaded and self.model is not None:
            # Real YOLOv11 inference
            results = self.model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                classes=classes,
                verbose=False
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = r.names[cls_id] if r.names else str(cls_id)
                    conf = float(box.conf[0].item())
                    # [x1, y1, x2, y2]
                    xyxy = [float(coord) for coord in box.xyxy[0].tolist()]

                    detections.append({
                        "class": cls_name,
                        "class_id": cls_id,
                        "confidence": round(conf, 4),
                        "bbox": xyxy
                    })

            # Extract timing breakdown from results if present
            speed = results[0].speed if results and hasattr(results[0], 'speed') else {}

            return {
                "status": "success",
                "model": os.path.basename(self.model_path),
                "device": str(self.model.device),
                "image_size": [image.width, image.height],
                "detections": detections,
                "count": len(detections),
                "latency_ms": round(latency_ms, 2),
                "speed_breakdown_ms": speed
            }
        else:
            # Fallback mock detector for testing when model file or ultralytics is not yet configured
            time.sleep(0.04) # Simulate 40ms inference
            latency_ms = (time.perf_counter() - start_time) * 1000
            w, h = image.size

            # Sample simulated detections
            mock_boxes = [
                {"class": "plastic bottle", "class_id": 39, "confidence": 0.93, "bbox": [w * 0.2, h * 0.25, w * 0.32, h * 0.55]},
                {"class": "plastic bag", "class_id": 40, "confidence": 0.87, "bbox": [w * 0.6, h * 0.3, w * 0.78, h * 0.6]},
                {"class": "cup", "class_id": 41, "confidence": 0.76, "bbox": [w * 0.4, h * 0.5, w * 0.48, h * 0.72]},
            ]
            filtered = [b for b in mock_boxes if b["confidence"] >= confidence]

            return {
                "status": "fallback_mock",
                "model": "yolo11-simulated",
                "device": "CPU (Simulated)",
                "image_size": [w, h],
                "detections": filtered,
                "count": len(filtered),
                "latency_ms": round(latency_ms, 2),
                "speed_breakdown_ms": {"preprocess": 1.2, "inference": round(latency_ms, 1), "postprocess": 0.8}
            }

# Global singleton
yolo_engine = YOLOv11Service()
