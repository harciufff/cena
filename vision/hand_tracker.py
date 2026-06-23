import cv2
import os
import sys
import time
import mediapipe as mp


# HandTracker uses MediaPipe to detect hand landmarks in video frames.
class HandTracker:
    # Initialize the hand tracker with optional model file name.
    # Set up internal state and load the hand‑landmark model.
    def __init__(self, model_name='hand_landmarker.task'):
        # Determine the correct path to load the model file.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_in_vision = os.path.join(current_dir, model_name)
        path_in_root = os.path.join(os.path.dirname(current_dir), model_name)

        # Check in the current directory, parent directory, or fallback to default.
        if os.path.exists(path_in_vision):
            self.model_path = path_in_vision
        elif os.path.exists(path_in_root):
            self.model_path = path_in_root
        else:
            self.model_path = model_name

        # List to store the hand tracking data from the most recently processed frame.
        self.latest_hands = []

        # Shorten names for MediaPipe Task API components.
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Try initializing the MediaPipe hand landmarker with GPU acceleration.
        try:
            base_options = BaseOptions(
                model_asset_path=self.model_path,
                delegate=BaseOptions.Delegate.GPU
            )
            print(f"MediaPipe: Loaded '{self.model_path}' with GPU (Metal) acceleration")
        except Exception as e:
            # Fall back to CPU if GPU initialization fails.
            print(f"GPU initialization error: {e}. Falling back to CPU...")
            base_options = BaseOptions(
                model_asset_path=self.model_path,
                delegate=BaseOptions.Delegate.CPU
            )

        # Configure detection and tracking parameters for a video stream.
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create the Landmarker instance using the configured options.
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        # Connection pairs representing bones/segments between hand landmarks.
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

    # Process a single video frame, detect hands, and draw landmarks on the frame.
    def process_frame(self, frame, timestamp_ms):
        # Create a copy of the frame to draw landmarks on without altering the original.
        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        # MediaPipe expects RGBA format; convert from OpenCV BGR format.
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)

        # Run hand landmark detection on the current video frame.
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        current_frame_hands = []

        # If any hands are detected, parse and render their landmarks.
        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                points = []
                landmarks_normalized = []

                # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates.
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append((cx, cy))
                    landmarks_normalized.append({"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)})

                # Store the hand index and normalized landmark coordinates.
                current_frame_hands.append({
                    "hand_index": hand_idx,
                    "landmarks": landmarks_normalized
                })

                # Draw connection lines between the landmarks.
                for connection in self.HAND_CONNECTIONS:
                    p1_idx, p2_idx = connection
                    if p1_idx < len(points) and p2_idx < len(points):
                        cv2.line(annotated_frame, points[p1_idx], points[p2_idx], (240, 240, 240), 2, cv2.LINE_AA)

                # Draw outer and inner circles for each joint/landmark.
                for pt in points:
                    cv2.circle(annotated_frame, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(annotated_frame, pt, 5, (0, 100, 0), 1, cv2.LINE_AA)

        # Update the list of recently detected hands.
        self.latest_hands = current_frame_hands

        # Return the annotated copy of the frame.
        return annotated_frame

    # Release and close the MediaPipe hand landmarker resource.
    def close(self):
        if self.landmarker:
            self.landmarker.close()