"""
Test script for the YOLOv11 FastAPI backend.
Run this script to verify your backend before connecting the dashboard.
Usage:
    python backend/test_client.py
"""

import sys
import requests
from io import BytesIO
from PIL import Image, ImageDraw

API_URL = "http://localhost:8000"

def test_health():
    print(f"--> Checking Health on {API_URL}/health ...")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status Code: {r.status_code}")
        print("Response:", r.json())
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"[!] Could not connect to {API_URL}. Ensure uvicorn is running:")
        print("    uvicorn backend.main:app --reload")
        return False

def test_prediction():
    print("\n--> Creating synthetic test image ...")
    img = Image.new("RGB", (640, 480), color=(30, 40, 60))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 250, 350], fill=(220, 50, 50))
    draw.rectangle([320, 180, 500, 300], fill=(50, 150, 220))

    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    print("--> Sending POST /predict/image ...")
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"confidence": 0.35, "iou": 0.45}

    r = requests.post(f"{API_URL}/predict/image", files=files, data=data)
    print(f"Status Code: {r.status_code}")
    print("Inference Result:", r.json())

if __name__ == "__main__":
    if test_health():
        test_prediction()
