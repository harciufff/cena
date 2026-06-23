import cv2
import numpy as np
import sys


def main():
    # Initialize the webcam using Apple's native AVFoundation backend for M-series chips.
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    # Verify if the webcam stream has been successfully opened.
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        sys.exit(1)

    # Create an auto-sizing OpenCV GUI window to display the consolidated dashboard.
    cv2.namedWindow("AI Pipeline Dashboard", cv2.WINDOW_AUTOSIZE)
    print("Dashboard started. Press 'q' to quit.")

    while True:
        # 1. Capture the raw video frame from the camera stream.
        ret, frame = cap.read()

        # Break the loop if the frame capture fails.
        if not ret:
            print("Error: Failed to grab a frame from the camera.")
            break

        # 2. Resize the frame so three of them fit nicely on a MacBook screen.
        # Scaling down by 50% is ideal for a 3-panel horizontal layout.
        h, w = frame.shape[:2]
        scale = 0.5
        resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # 3. Create independent copies for our 3 sections.
        # (Once we add YOLO and MediaPipe, they will draw on their respective frames here)
        frame_clean = resized_frame.copy()
        frame_yolo = resized_frame.copy()
        frame_mp = resized_frame.copy()

        # 4. Add text labels to each frame so we know which is which.
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)  # Green text color in BGR format
        thickness = 2
        cv2.putText(frame_clean, "Clean Frame", (20, 40), font, 1, color, thickness)
        cv2.putText(frame_yolo, "YOLO-World", (20, 40), font, 1, color, thickness)
        cv2.putText(frame_mp, "MediaPipe Hands", (20, 40), font, 1, color, thickness)

        # 5. Concatenate the frames horizontally into one single wide image.
        # You could also use np.vstack for vertical, or a mix for a 2x2 grid.
        combined_dashboard = np.hstack((frame_clean, frame_yolo, frame_mp))

        # 6. Render the single dashboard window.
        cv2.imshow("AI Pipeline Dashboard", combined_dashboard)

        # 7. Handle UI events and check if the 'q' key was pressed to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received. Shutting down...")
            break

    # Release the camera resource and destroy all OpenCV GUI windows.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()