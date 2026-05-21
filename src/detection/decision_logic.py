from ultralytics import YOLO
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_FILE = BASE_DIR / "results" / "log.txt"
import sys

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

        print(f"{image_path.name} -> persons: {person_count}, vehicles: {vehicle_count}")

        msg = llm_decision(person_count, vehicle_count, image_path.name)

        print(f"[LLM] {msg}")
        print("DEBUG: CALLING save_log")
        save_log(f"{image_path.name} -> {msg}")

        if "ALERT" in msg:
             send_alert(msg)

        print("-" * 40)

if __name__ == "__main__":
        main()