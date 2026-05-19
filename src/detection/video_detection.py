import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from pathlib import Path
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

def main():
    for video_path in video_dir.glob("*.mp4"):

        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0

        print(f"\nProcessing video: {video_path.name}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 10 != 0:
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

            msg = llm_decision(person_count, vehicle_count, video_path.name)

            print(f"[Frame {frame_count}] persons={person_count}, vehicles={vehicle_count}")
            print(f"[LLM] {msg}")

            if "ALERT" in msg:
                send_alert(msg)

        cap.release()