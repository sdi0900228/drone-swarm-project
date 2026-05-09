from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

input_dir = Path("data/samples")

target_classes = {"person", "car", "bus", "truck"}

image_extensions = {".jpg", ".jpeg", ".png"}
image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]

for image_path in image_files:
    results = model(image_path)

    for r in results:
        names = r.names
        detected = []

        for cls_id in r.boxes.cls:
            class_name = names[int(cls_id)]
            if class_name in target_classes:
                detected.append(class_name)

        print(f"{image_path.name} -> {detected}")