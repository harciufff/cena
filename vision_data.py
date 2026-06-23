import time
import json
import os
from dotenv import load_dotenv

load_dotenv()


class VisionDataAggregator:
    # Orchestrator that merges Semantics (Slow) with Geometry (Fast)
    # and writes the results directly to Obsidian-compatible Markdown files.
    def __init__(self, vlm_worker, yolo_detector, hand_tracker):
        self.vlm = vlm_worker
        self.yolo = yolo_detector
        self.tracker = hand_tracker

        # Obsidian folder configuration
        self.obsidian_dir = os.getenv("OBSIDIAN_VAULT_PATH")

        if not self.obsidian_dir:
            raise ValueError("Error: OBSIDIAN_VAULT_PATH is not configured in the .env file!")

        self.live_file_path = os.path.join(self.obsidian_dir, "LiveContext.md")
        self.persistent_file_path = os.path.join(self.obsidian_dir, "PersistentContext.md")

        # Ensure the CENA directory exists before trying to write files.
        os.makedirs(self.obsidian_dir, exist_ok=True)
        print(f"VisionDataAggregator: Connected to Obsidian folder: {self.obsidian_dir}")

        # Debug: verify if the target attributes exist on the initialized objects.
        print(f"DEBUG: Does YoloDetector have 'latest_detections'? {hasattr(self.yolo, 'latest_detections')}")
        print(f"DEBUG: Does HandTracker have 'latest_hands'? {hasattr(self.tracker, 'latest_hands')}")

    # Called every frame in main.py.
    # If VlmAnalyzer produced a new description, it merges the data and updates the Obsidian files.
    def check_and_aggregate(self):
        # Check if there is new text ready in a thread-safe manner.
        with self.vlm.lock:
            is_ready = self.vlm.new_text_ready or self.vlm.latest_text is not None

        if is_ready:
            # 1. Retrieve the slow thinking data (Semantics).
            with self.vlm.lock:
                current_semantics = self.vlm.latest_text if self.vlm.latest_text else "No description available"
                # Reset the new text availability flag.
                self.vlm.new_text_ready = False

            # 2. Retrieve the fast geometry data (with DEBUG checks).
            yolo_has_attr = hasattr(self.yolo, 'latest_detections')
            tracker_has_attr = hasattr(self.tracker, 'latest_hands')

            current_objects = self.yolo.latest_detections if yolo_has_attr else []
            current_hands = self.tracker.latest_hands if tracker_has_attr else []

            # Debug: show how much data we have collected for this snapshot.
            print(f"DEBUG check_and_aggregate: YOLO Objects={len(current_objects)}, Hands={len(current_hands)}")
            if current_objects:
                print(f"  → Objects: {current_objects[:3]}")  # Show the first 3 objects
            if current_hands:
                print(f"  → Hands: {len(current_hands)} hands detected")

            # 3. Create the Unified Record snapshot combining all vision models' outputs.
            snapshot = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "semantics": current_semantics,
                "geometry": {
                    "detected_objects": current_objects,
                    "hand_landmarks": current_hands
                }
            }

            # 4. Write simultaneously to both Obsidian Markdown files.
            self._write_to_obsidian(snapshot)

            # Reset the latest text to prevent duplicate writes on the next frame.
            with self.vlm.lock:
                self.vlm.latest_text = None

            return snapshot

        return None

    # Writes the snapshot to both the LiveContext and PersistentContext files.
    def _write_to_obsidian(self, snapshot):
        timestamp = snapshot["timestamp"]
        semantics = snapshot["semantics"]
        objects = snapshot["geometry"]["detected_objects"]
        hands = snapshot["geometry"]["hand_landmarks"]

        # Generate the Markdown content for the live dashboard.
        markdown_content = f"""# CENA
*Last update: {timestamp}*


## Semantic Analysis (Moondream)
> {semantics}


## Geometric Analysis


### Detected Objects (YOLO-World)
"""
        # Format the detected objects into a Markdown table.
        if objects and len(objects) > 0:
            markdown_content += "| Object | Confidence | Position (Bounding Box) |\n| :--- | :---: | :--- |\n"
            for obj in objects:
                if isinstance(obj, dict):
                    name = obj.get('label', 'Unknown')
                    conf = f"{obj.get('confidence', 0.0):.2f}"
                    bbox = obj.get('box', 'N/A')
                else:
                    name = obj[0] if len(obj) > 0 else 'Unknown'
                    conf = f"{obj[1]:.2f}" if len(obj) > 1 else '0.00'
                    bbox = obj[2] if len(obj) > 2 else 'N/A'
                markdown_content += f"| **{name}** | {conf} | `{bbox}` |\n"
        else:
            markdown_content += "_No objects detected at the moment._\n"

        # Format the hand tracking data into an expandable JSON block.
        markdown_content += "\n### Hand Tracking (MediaPipe)\n"
        if hands and len(hands) > 0:
            markdown_content += f"**Detected {len(hands)} active hands.**\n"
            markdown_content += "<details>\n<summary>Show Landmark coordinates</summary>\n\n"
            markdown_content += "```json\n" + json.dumps(hands, indent=2) + "\n```\n</details>\n"
        else:
            markdown_content += "_No hands detected in the field of view._\n"

        markdown_content += "\n---\n"

        # Write the formatted content to liveContext.md, overwriting previous data.
        try:
            with open(self.live_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
        except Exception as e:
            print(f"[Error writing LiveContext.md] {e}")

        # Prepare and append a historical log entry to persistentContext.md.
        objects_names = []
        for obj in objects if objects else []:
            if isinstance(obj, dict):
                objects_names.append(obj.get('label', 'Unknown'))
            else:
                objects_names.append(obj[0] if len(obj) > 0 else 'Unknown')

        persistent_entry = f"""## Log of {timestamp}
* **Semantics:** {semantics}
* **Objects:** {', '.join(objects_names) if objects_names else 'None'}
* **Hands:** {'Detected (' + str(len(hands)) + ')' if hands and len(hands) > 0 else 'None'}


"""
        # Write to the persistent log file, creating the header if it doesn't exist yet.
        try:
            file_exists = os.path.exists(self.persistent_file_path)
            with open(self.persistent_file_path, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write("# CENA (Persistent Log)\n\n")
                f.write(persistent_entry)
        except Exception as e:
            print(f"[Error writing PersistentContext.md] {e}")

        # Feedback on the terminal confirming the write operation.
        objs_count = len(objects) if objects else 0
        hands_count = len(hands) if hands else 0
        print(
            f"[-> OBSIDIAN @ {time.strftime('%H:%M:%S')}] Semantics: '{semantics[:50]}...' | YOLO Objects: {objs_count} | MediaPipe Hands: {hands_count}")