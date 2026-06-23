import cv2
import os
import sys
from ultralytics import YOLOWorld


class YoloDetector:
    # Initialize YOLO-World Large optimized for Apple Silicon (M4).
    def __init__(self, model_name='yolov8x-worldv2.pt'):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.pt_path = os.path.join(current_dir, model_name)
        self.mlpackage_path = os.path.join(current_dir, model_name.replace('.pt', '.mlpackage'))

        # --- ADDED FOR AGGREGATOR ---
        # This list will store the detected object data from the latest frame.
        self.latest_detections = []

        # Check if a compiled CoreML model (.mlpackage) exists for Apple NPU execution.
        if os.path.exists(self.mlpackage_path):
            print("YOLO-World: Compiled CoreML model detected. Starting on NPU...")
            self.model = YOLOWorld(self.mlpackage_path)
            self.device = 'cpu'
        else:
            # Fall back to PyTorch with Metal Performance Shaders (MPS) GPU acceleration.
            print("YOLO-World: Starting PyTorch model with GPU (MPS) acceleration...")
            self.model = YOLOWorld(model_name)
            self.device = 'mps'

        try:
            # Configure the default LVIS vocabulary (supporting over 1200 object classes).
            print("YOLO-World: Configuring LVIS vocabulary (1200+ objects)...")
            self.model.set_classes(None)
        except Exception as e:
            print(f"Class configuration note: {e}")

    # Run inference, draw bounding boxes, and update self.latest_detections.
    def process_frame(self, frame):
        annotated_frame = frame.copy()

        # --- ADDED FOR AGGREGATOR ---
        current_frame_detections = []

        # Run object detection inference on the frame.
        results = self.model.predict(
            frame,
            device=self.device,
            verbose=False,
            conf=0.25,
            imgsz=640
        )

        # Parse results and format bounding boxes if detections are found.
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]

                # Save the object details in a structured dictionary.
                current_frame_detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "box": [x1, y1, x2, y2]
                })

                # Drawing graphic overlays (bounding boxes and labels) on the frame.
                color = (0, 140, 255)  # Orange border color in BGR format
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                caption = f"{label} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                # Draw a filled background rectangle behind the text label for better legibility.
                (text_w, text_h), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
                cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
                cv2.putText(annotated_frame, caption, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness,
                            cv2.LINE_AA)

        # Update the class state with the latest frame's detections.
        self.latest_detections = current_frame_detections

        return annotated_frame