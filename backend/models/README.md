# YOLOv11 Model Weights Directory

Place your trained YOLOv11 model weights (`.pt` or `.onnx` files) in this directory:

### Supported Model Formats:
1. **Pretrained Ultralytics YOLOv11 weights:**
   - `yolo11n.pt` (Nano - Ultra fast, best for edge & web)
   - `yolo11s.pt` (Small - Balanced accuracy & speed)
   - `yolo11m.pt` (Medium - High accuracy)
   - `yolo11l.pt` (Large - Production grade)
   - `yolo11x.pt` (Extra Large - Maximum mAP)

2. **Custom Trained Weights:**
   - Drop your custom trained model weights here (e.g. `best.pt`, `ocean_plastics_yolo11.pt`).

### Automatic Download:
When you install `ultralytics` and run `YOLO('yolo11n.pt')`, the official weights are downloaded automatically by the library!

To export to ONNX for even faster inference:
```bash
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); model.export(format='onnx')"
```
