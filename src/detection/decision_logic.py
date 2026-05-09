from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")

input_dir = Path("data/samples")

image_extensions = {".jpg", ".jpeg", ".png"}
image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]

for image_path in image_files:
    results = model(image_path)

    person_count = 0
    vehicle_count = 0

    for r in results:
        names = r.names

        for cls_id in r.boxes.cls:
            class_name = names[int(cls_id)]

            if class_name == "person":
                person_count += 1
            elif class_name in {"car", "bus", "truck"}:
                vehicle_count += 1

    print(f"{image_path.name} -> persons: {person_count}, vehicles: {vehicle_count}")

    if person_count > 5:
        print("ALERT: crowd detected")

    if vehicle_count > 10:
        print("ALERT: heavy traffic detected")

    if person_count > 0 and vehicle_count > 0:
        print("INFO: mixed activity detected")

    print("-" * 40)