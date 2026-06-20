from ultralytics import YOLO
from pathlib import Path

import os
import random
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_FILE = BASE_DIR / "results" / "log.txt"

sys.path.append(str(Path(__file__).resolve().parent.parent))

from communication.mqtt_publisher import send_alert

model = YOLO("yolov8n.pt")

input_dir = Path("data/samples")

image_extensions = {".jpg", ".jpeg", ".png"}
image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]

def save_log(message):
    os.makedirs(BASE_DIR / "results", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")
    print("LOG PATH:", LOG_FILE)

def llm_decision(persons, vehicles, source):

    crowd_msgs = [
        f"ALERT: High crowd density detected in {source}. The area appears heavily populated.",
        f"ALERT: Large number of people observed in {source}. Potential crowd formation.",
        f"ALERT: Crowd activity is high in {source}. Monitoring recommended."
    ]

    traffic_msgs = [
        f"ALERT: Heavy traffic detected in {source}. Possible congestion.",
        f"ALERT: Significant vehicle presence in {source}. Traffic conditions may be critical.",
        f"ALERT: Increased traffic levels observed in {source}."
    ]

    mixed_msgs = [
        f"INFO: Mixed activity detected in {source} (people and vehicles).",
        f"INFO: Both pedestrian and vehicle presence observed in {source}.",
        f"INFO: Combined movement detected in {source}."
    ]

    normal_msgs = [
        f"OK: No significant activity detected in {source}.",
        f"OK: Area appears calm in {source}.",
        f"OK: No unusual patterns detected in {source}."
    ]

    if persons > 5:
        return random.choice(crowd_msgs)

    elif vehicles > 10:
        return random.choice(traffic_msgs)

    elif persons > 0 and vehicles > 0:
        return random.choice(mixed_msgs)

    else:
        return random.choice(normal_msgs)

def main():
    for image_path in image_files:
        print(f"[INFO] Processing: {image_path.name}")

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

        print(f"[RESULT] {image_path.name} -> persons: {person_count}, vehicles: {vehicle_count}")

        msg = llm_decision(person_count, vehicle_count, image_path.name)

        print(f"[LLM] {msg}")
        save_log(f"{image_path.name} -> {msg}")

        if "ALERT" in msg:
             send_alert(msg)

        print("=" * 40)

if __name__ == "__main__":
    main()