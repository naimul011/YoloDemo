from ultralytics import YOLO
model = YOLO(r'.\runs\detect\train20\weights\best.pt')
results = model.val()
