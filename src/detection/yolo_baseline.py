from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

input_dir = Path("data/samples")
output_dir = Path("results/detections")

image_extensions = {".jpg", ".jpeg", ".png"}
image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]

if not image_files:
    print("No sample images found in data/samples")
    raise SystemExit(1)

for image_path in image_files:
    model.predict(source=str(image_path), save=True, project=str(output_dir), name="baseline", exist_ok=True)
    print(f"Processed: {image_path.name}")

print("Baseline YOLO test completed for all sample images.")
print(f"Results saved under: {output_dir / 'baseline'}")