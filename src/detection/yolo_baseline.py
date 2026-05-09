from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

input_path = Path("data/samples/test1.jpg")
output_dir = Path("results/detections")

results = model.predict(source=str(input_path), save=True, project=str(output_dir), name="baseline", exist_ok=True)

print("Baseline YOLO test completed.")
print(f"Input image: {input_path}")
print(f"Results saved under: {output_dir / 'baseline'}")