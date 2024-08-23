from ultralytics import YOLO
model = YOLO(r'.\runs\detect\train20\weights\best.pt')

results = model(r'.\dataset\valid\images\20161017_000000_1024_HMIIF_jpg.rf.58066929d27b28cbef80b00f03f7012d.jpg',save=True)

print(results)
