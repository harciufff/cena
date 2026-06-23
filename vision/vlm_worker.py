import cv2
import time
import threading
from PIL import Image
import moondream as md
import os
import ssl
import warnings
from dotenv import load_dotenv

load_dotenv()

# Set the Hugging Face token required by Moondream.
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Ignore non-critical system warnings so the console output remains focused on actual errors.
warnings.filterwarnings("ignore", category=UserWarning)

# Disable tokenizers parallelism to prevent warning messages when using threads.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Bypass SSL verification to avoid macOS local development issues during model downloads.
ssl._create_default_https_context = ssl._create_unverified_context


class VlmAnalyzer:
    # Initialize the analyzer state for the background Vision Language Model worker.
    # The 'interval' parameter sets the delay (in seconds) between Moondream queries.
    def __init__(self, api_key="", interval=10):
        self.interval = interval
        self.latest_frame = None
        self.running = False

        # Thread lock used to prevent race conditions when reading/writing the latest frame.
        self.lock = threading.Lock()

        self.latest_text = ""
        self.new_text_ready = False

        print(f"VLM: Loading Moondream model locally with Metal acceleration...")

        # Try to load the Moondream vision language model through its official API.
        try:
            self.model = md.vl(api_key=api_key, local=True)
            print("VLM: Moondream loaded successfully")
        except Exception as e:
            # If the model fails to load, print the error and exit the program.
            print(f"Moondream initialization error: {e}")
            exit()

        # Moondream automatically utilizes the MPS (Metal Performance Shaders) backend on Apple Silicon.
        self.device = "mps"

    # Store a copy of the most recent webcam frame.
    # This ensures the background worker always has the newest image to process.
    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    # Start the daemon thread that will repeatedly run the analysis loop.
    # Daemon threads automatically exit when the main program finishes.
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        print(f"VLM: Async thread active. Analysis runs every {self.interval} seconds.")

    # Signal the background worker thread to stop and wait for it to cleanly terminate.
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    # The core background loop running on an independent thread.
    # It waits, grabs the latest frame, and sends it to the Moondream model for analysis.
    def _worker_loop(self):
        prompt = "Describe the scene briefly and clearly."

        while self.running:
            # Pause the thread for the specified interval.
            time.sleep(self.interval)

            # Safely grab a copy of the latest frame.
            with self.lock:
                if self.latest_frame is None:
                    continue
                frame_to_process = self.latest_frame.copy()

            # OpenCV uses BGR natively, but the model (and PIL) expects RGB.
            rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            try:
                # Encode the PIL image and query the Moondream model with the text prompt.
                encoded_image = self.model.encode_image(pil_image)
                result = self.model.query(encoded_image, prompt)

                # Extract and clean up the text response.
                final_text = result["answer"].strip()

                # Safely update the shared state variables with the new description.
                with self.lock:
                    self.latest_text = final_text
                    self.new_text_ready = True

            except Exception as e:
                # Catch and print any runtime errors occurring during model inference.
                print(f"\n[Moondream Runtime Error] {e}\n")


# Entry point for testing the module independently.
if __name__ == "__main__":
    MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")

    if not MOONDREAM_API_KEY:
        print("Error: MOONDREAM_API_KEY not found. Did you create the .env file?")
        sys.exit(1)

    # Initialize and start the asynchronous VLM analyzer.
    vlm = VlmAnalyzer(api_key=MOONDREAM_API_KEY, interval=10)
    vlm.start()

    # Open the default webcam stream.
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Send the newest frame to the analyzer and display it on screen.
        vlm.update_frame(frame)
        cv2.imshow("Webcam", frame)

        # Quit the test loop if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Safely shut down all background processes and close OpenCV windows.
    vlm.stop()
    cap.release()
    cv2.destroyAllWindows()