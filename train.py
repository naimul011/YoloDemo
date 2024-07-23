from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  

model.train(data='data.yaml', epochs=2, imgsz=640)
