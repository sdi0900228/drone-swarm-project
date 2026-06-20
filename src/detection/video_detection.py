import cv2
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from communication.mqtt_publisher import send_alert

model = YOLO("yolov8n.pt")

video_dir = Path("data/videos")

def llm_decision(persons, vehicles, frame_name):
    if persons > 5:
        return f"ALERT: High crowd density in {frame_name}"
    elif vehicles > 10:
        return f"ALERT: Heavy traffic in {frame_name}"
    elif persons > 0 and vehicles > 0:
        return f"INFO: Mixed activity in {frame_name}"
    else:
        return f"OK: No significant activity in {frame_name}"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_FILE = BASE_DIR / "results" / "log.txt"

def save_log(message):
    os.makedirs(BASE_DIR / "results", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

FRAME_SKIP = 10

def main():
    for video_path in video_dir.glob("*.mp4"):

        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        last_person_count = 0
        last_msg = None

        print(f"\nProcessing video: {video_path.name}")
        print(f"[INFO] Using frame skip: {FRAME_SKIP}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % FRAME_SKIP != 0:
                continue

            results = model(frame)

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

            if person_count > last_person_count:
                trend = "increasing"
            elif person_count < last_person_count:
                trend = "decreasing"
            else:
                trend = "stable"

            last_person_count = person_count

            msg = llm_decision(person_count, vehicle_count, video_path.name)

            msg = msg + f" (trend: {trend})"

            print(f"[Frame {frame_count}] persons={person_count}, vehicles={vehicle_count}")
            print(f"[LLM] {msg}")

            if msg != last_msg:

                print(f"[STATE CHANGE] {msg}")
                save_log(f"{video_path.name} [frame {frame_count}] -> {msg}")

                if "ALERT" in msg:
                    send_alert(msg)

                last_msg = msg

        cap.release()


if __name__ == "__main__":
    main()