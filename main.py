from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

def run_images():
    print("\n[MODE] Image Detection\n")
    from src.detection.decision_logic import main as image_main
    image_main()

def run_videos():
    print("\n[MODE] Video Detection\n")
    from src.detection.video_detection import main as video_main
    video_main()

def main():
    print("===================================")
    print(" AI Drone Swarm Monitoring System ")
    print("===================================")

    print("\nSelect mode:")
    print("1. Image Detection")
    print("2. Video Detection")

    choice = input("\nEnter choice (1 or 2): ")

    if choice == "1":
        run_images()
    elif choice == "2":
        run_videos()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()