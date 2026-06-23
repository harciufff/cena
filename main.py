import cv2
import numpy as np
import sys
import time
import os
import ssl
import warnings
import threading
from PIL import Image
import moondream as md
from dotenv import load_dotenv

load_dotenv()

# Disable unnecessary warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set HF token for Moondream local download
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Bypass macOS SSL for local model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Import components from vision package
from vision import HandTracker, YoloDetector
from vision_data import VisionDataAggregator
from vision.vlm_worker import VlmAnalyzer


def main():
    # Initialize webcam with AVFoundation (macOS)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        sys.exit(1)

    # Load Moondream API key from environment
    MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")

    if not MOONDREAM_API_KEY:
        print("Error: MOONDREAM_API_KEY not found. Did you create the .env file?")
        sys.exit(1)

    # Initialize Moondream VLM (async background thread)
    try:
        vlm = VlmAnalyzer(api_key=MOONDREAM_API_KEY, interval=10)
        vlm.start()
    except Exception as e:
        print(f"Critical error starting Moondream: {e}")
        cap.release()
        sys.exit(1)

    # Initialize Hand Tracker (GPU)
    try:
        tracker = HandTracker(model_name='hand_landmarker.task')
    except Exception as e:
        print(f"Critical error starting HandTracker: {e}")
        vlm.stop()
        cap.release()
        sys.exit(1)

    # Initialize YOLO-World detector (GPU)
    try:
        detector = YoloDetector(model_name='yolov8x-worldv2.pt')
    except Exception as e:
        print(f"Error starting YOLO-World: {e}")
        vlm.stop()
        tracker.close()
        cap.release()
        sys.exit(1)

    # Initialize Obsidian data aggregator
    try:
        aggregator = VisionDataAggregator(
            vlm_worker=vlm,
            yolo_detector=detector,
            hand_tracker=tracker
        )
    except Exception as e:
        print(f"Error starting VisionDataAggregator: {e}")
        vlm.stop()
        tracker.close()
        cap.release()
        sys.exit(1)

    # Create auto-sizing window for 3-panel dashboard
    cv2.namedWindow("CENA", cv2.WINDOW_AUTOSIZE)

    print("  CENA Active! Moondream, YOLO & Hands Ready.  ")
    print("  Press 'q' to quit the program.        ")

    prev_time = time.perf_counter()

    # Main frame processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No frame received from the camera.")
            break

        # Calculate system FPS
        curr_time = time.perf_counter()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        timestamp_ms = int(curr_time * 1000)

        # Scale frame by 50% for 3-panel layout
        h, w = frame.shape[:2]
        scale = 0.5
        resized_w = int(w * scale)
        resized_h = int(h * scale)
        resized_frame = cv2.resize(frame, (resized_w, resized_h))

        # Panel 1: Clean frame for Moondream
        panel_clean = resized_frame.copy()
        vlm.update_frame(panel_clean)
        aggregator.check_and_aggregate()

        # Panel 2: YOLO-World detections
        panel_yolo = detector.process_frame(resized_frame)

        # Panel 3: MediaPipe hand tracking
        panel_mp = tracker.process_frame(resized_frame, timestamp_ms)

        # Add panel labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color_green = (0, 255, 0)
        color_info = (255, 120, 0)
        thickness = 2

        cv2.putText(panel_clean, "1. Clean Frame (Moondream2)", (15, 35), font, font_scale, color_green, thickness)
        cv2.putText(panel_yolo, "2. YOLO-World (Detections)", (15, 35), font, font_scale, color_green, thickness)
        cv2.putText(panel_mp, "3. MediaPipe (Hands Tracker)", (15, 35), font, font_scale, color_green, thickness)

        # Display FPS
        cv2.putText(panel_clean, f"System FPS: {int(fps)}", (15, resized_h - 15), font, 0.5, color_info, 1)

        # Combine panels horizontally
        dashboard_image = np.hstack((panel_clean, panel_yolo, panel_mp))
        cv2.imshow("CENA", dashboard_image)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    print("Shutting down, please wait...")
    vlm.stop()
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("Pipeline successfully stopped.")


if __name__ == "__main__":
    main()