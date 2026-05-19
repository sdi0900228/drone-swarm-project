from ultralytics import YOLO
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from communication.mqtt_publisher import send_alert

model = YOLO("yolov8n.pt")

input_dir = Path("data/samples")

image_extensions = {".jpg", ".jpeg", ".png"}
image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]


def llm_decision(persons, vehicles, image_name):
    """
    Simulated LLM reasoning (natural language output)
    """
    if persons > 5:
        return f"ALERT: High crowd density detected. The area appears crowded. Source: {image_name}"
    elif vehicles > 10:
        return f"ALERT: Heavy traffic detected. Possible congestion. Source: {image_name}"
    elif persons > 0 and vehicles > 0:
        return f"INFO: Mixed activity detected (people and vehicles). Source: {image_name}"
    else:
        return f"OK: No significant activity detected. Source: {image_name}"


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

            elif class_name in {"car", "bus", "truck", "motorcycle", "bicycle"}:
                vehicle_count += 1

    print(f"{image_path.name} -> persons: {person_count}, vehicles: {vehicle_count}")

    msg = llm_decision(person_count, vehicle_count, image_path.name)

    print(f"[LLM] {msg}")

    if "ALERT" in msg:
        send_alert(msg)

    print("-" * 40)