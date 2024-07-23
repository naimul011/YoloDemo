from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # You can also use yolov8s.yaml, yolov8m.yaml, etc.

model.train(data='data.yaml', epochs=2, imgsz=640)
