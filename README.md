# 🚀 YOLOv11 Vision Hub & Telemetry Dashboard

> The TalentRank AI recruitment workspace is included separately in [`talentrank-ai/`](./talentrank-ai/), preserving this YOLOv11 application.

A modern, high-performance, real-time Computer Vision dashboard designed specifically for **YOLOv11** object detection, segmentation, and analytics.

Featuring both an **instant zero-install standalone dashboard** (`index.html`) and a **modular full-stack application** (React + Vite + Tailwind CSS frontend with a FastAPI + Ultralytics YOLOv11 backend).

---

## 📸 Key Features

- **⚡ Instant Testing (Simulator Mode)**: Test all UI features, canvas bounding boxes, webcam streams, and analytics right away without needing the model weights installed yet.
- **🔌 Pluggable YOLOv11 Backend**: Seamlessly connects to FastAPI running `ultralytics.YOLO('yolo11n.pt')` or custom-trained weights (`best.pt`).
- **📹 Dual Input Modes**:
  - **Image Upload & Presets**: Drag and drop any image or test with built-in scenarios (Ocean Plastics Waste, Traffic, Street).
  - **Live Camera / Webcam**: Real-time webcam inference with live FPS counter and low latency.
- **🎛️ Real-Time Hyperparameter Tuning**:
  - Dynamic **Confidence Threshold** slider (0.05 – 0.95)
  - Dynamic **IoU / NMS Threshold** slider (0.10 – 0.90)
  - Model architecture switcher (Nano, Small, Medium, Custom)
- **📊 Real-time Telemetry & Analytics**:
  - Live FPS gauge & forward latency breakdown
  - Class frequency distribution bars
  - Bounding box telemetry table
  - **Export Detections to JSON and CSV** with one click.

---

## 🏗️ Project Architecture

```
dashboard/
├── index.html                  # ⚡ Instant Standalone Dashboard (Runs directly in any browser)
├── .gitignore                  # Git ignore rules for node_modules, .venv, *.pt weights
├── README.md                   # Project documentation & GitHub push guide
│
├── frontend/                   # 🎨 Full-Stack React + Vite + Tailwind Frontend
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── index.css
│
└── backend/                    # 🐍 FastAPI + Ultralytics YOLOv11 Backend
    ├── main.py                 # FastAPI server (REST & WebSocket endpoints)
    ├── yolo_service.py         # YOLOv11 inference engine with fallback mock
    ├── requirements.txt        # Python dependencies
    ├── test_client.py          # Quick verification script
    └── models/
        └── README.md           # Instructions for placing yolo11*.pt / best.pt weights
```

---

## 🚀 Quick Start Guide

### Option A: Instant Browser Preview (Zero Installation)
Simply double click or open **`index.html`** in any web browser!
- Test the interactive dashboard immediately.
- Try the sample presets (Ocean Plastics, Traffic, Street).
- Test with your local webcam or upload your own images.
- Sliders and bounding box visualizers work out of the box with simulated YOLOv11 predictions.

---

### Option B: Full-Stack Setup (FastAPI + YOLOv11 Backend)

#### 1. Setup & Run the YOLOv11 Backend
```bash
# Navigate to backend and install requirements
pip install -r backend/requirements.txt

# Start the FastAPI inference server
uvicorn backend.main:app --reload --port 8000
```
The API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

#### 2. Run the React Frontend (Optional)
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the React application.

---

## 🎯 How to Connect Your Custom YOLOv11 Model

1. **Place your trained `.pt` weights**:
   Copy your trained weights file (e.g. `best.pt` or `ocean_plastics.pt`) into the `backend/models/` folder:
   ```bash
   backend/models/best.pt
   ```

2. **Update the model in `backend/yolo_service.py`**:
   ```python
   yolo_engine = YOLOv11Service(model_path="backend/models/best.pt")
   ```

3. **In the Dashboard**:
   - Click the **API Config** button in the header.
   - Set the API URL to `http://localhost:8000`.
   - Click **Test Health** and **Save Settings**.
   - Your dashboard is now running live inferences with your custom YOLOv11 model!

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check, device info, and loaded model status |
| `POST` | `/predict/image` | Runs detection on uploaded image; returns boxes & metrics |
| `POST` | `/model/load` | Dynamically switches loaded weights file |
| `WS` | `/ws/stream` | WebSocket endpoint for high-speed live camera streams |

---

## 🐙 How to Push this Project to GitHub

Follow these steps in your terminal inside the `dashboard` folder:

```bash
# 1. Initialize Git repository (if not already initialized)
git init

# 2. Stage all project files
git add .

# 3. Create initial commit
git commit -m "Initial commit: YOLOv11 Vision Dashboard & FastAPI backend"

# 4. Set default branch to main
git branch -M main

# 5. Add your GitHub remote repository (replace with your repo URL)
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git

# 6. Push to GitHub
git push -u origin main
```

---

## 📄 License
MIT License. Feel free to use and customize for your research, academic, and production projects!
