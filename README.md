# cena

![Experimental](https://img.shields.io/badge/Experimental-yellow?style=flat-square)

**Edge‑Native Visual Copilot** – a proof‑of‑concept multimodal system that enables high‑precision manual tasks through real‑time geometric telemetry and asynchronous semantic understanding. It couples **YOLO‑World** object detection, **MediaPipe Hands** hand‑tracking, and the lightweight Vision‑Language Model **Moondream‑2** to provide a continuously refreshed scene description without relying on a local vector database.

---

## Architecture & Pipeline

The system is organized into three loosely‑coupled layers that operate at different frame‑rates and never block each other:

1. **Geometric Layer (≈30 FPS)** – Captures webcam frames via OpenCV, feeds them in parallel to:
   * **YOLO‑World** for 1200‑class object detection.
   * **MediaPipe Hands** (GPU‑accelerated Metal) for 21‑point hand landmark tracking.
   This layer runs synchronously in the main loop and produces annotated frames for visual feedback.

2. **Semantic Layer (~150 ms per inference (called every 10s))** – A background thread invokes **Moondream‑2** every ~10 seconds on the latest frame. The VLM extracts a natural‑language description of the scene (objects, actions, relations) and writes the result into a thread‑safe buffer.

3. **Agent Layer** – The `VisionDataAggregator` merges the fast geometric data with the latest semantic text, then writes two markdown logs:
   * `LiveContext.md` – a sliding‑window snapshot refreshed every 10 seconds, representing the current world state.
   * `PersistentContext.md` – an append‑only chronicle that can be queried later.

The three layers communicate via lightweight Python queues and shared objects; no layer waits on another, guaranteeing real‑time responsiveness even during expensive VLM inference.

---

## The Dual‑File Markdown Memory

Inspired by Karpathy’s *LLM‑as‑an‑OS* concept, the system stores its external context in plain markdown rather than a vector store. This design offers two key benefits on edge devices:

* **Zero indexing latency** – writing to a text file is orders of magnitude faster than embedding generation and vector insertion, which is crucial when the device has limited GPU/CPU headroom.
* **Token‑efficient retrieval** – downstream agents (e.g., OpenClaw) can `grep` or stream‑read the markdown directly, avoiding the token‑overhead of a dense‑vector lookup.

`LiveContext.md` works as a short‑term “window” that always reflects the latest perception, while `PersistentContext.md` provides a durable audit trail for post‑hoc analysis, debugging, or long‑term memory.

---

## Agent Integration

External agents read the markdown files natively:

* **OpenClaw** – mount the folder as a Graph‑RAG source. The agents treat each line as a knowledge node, allowing instant, up‑to‑date grounding for voice commands or UI actions.
* **CLI tools** – can `cat` or `tail -f` the live file to obtain the current world description without any additional API.

Because the format is plain markdown, integration requires only a file‑watcher; no custom SDK or network service is needed, which aligns perfectly with edge‑first privacy constraints.

---

## Current Status & Hardware Bottlenecks

| Component | Observed performance on Mac M4 (24 GB RAM) | Notes |
|-----------|----------------------------------------|-------|
| YOLO‑World (CoreML) | ~30 FPS (GPU/NPU) | Runs efficiently on Apple silicon NPU; fallback to PyTorch‑MPS if CoreML model unavailable.
| MediaPipe Hands | ~30 FPS (GPU) | Low latency hand landmark extraction.
| Moondream‑2 (Metal) | ~150 ms per inference (called every 10s) | The VLM is the primary thermal hotspot; prolonged runs cause CPU‑GPU throttling after ~5 minutes.
| File I/O (markdown) | < 5 ms per write | Negligible.

**Limitations** – The current proof‑of‑concept is limited to a single‑device setup. For industrial‑scale deployments (continuous 24 h operation, multiple concurrent streams) the Moondream inference would benefit from a custom‑fine‑tuned model or a dedicated accelerator (e.g., Apple Max/Ultra chip) to avoid thermal throttling.

---

## Installation & Requirements

1. **Python 3.10+** (tested on 3.11).
2. **Virtual environment** – `python -m venv venv && source venv/bin/activate`.
3. **Dependencies** – `pip install -r requirements.txt`.
4. **Environment variables** (see `.env.example`):
    * `HF_TOKEN` – HuggingFace access token. Required to securely download the **Moondream‑2** model weights from the HuggingFace Hub. To get one: Log into [HuggingFace](https://huggingface.co/) -> Settings -> Access Tokens -> Generate a new "Read" token.
    * `MOONDREAM_API_KEY` – Required by the Moondream library to validate the runtime and suppress initialization warnings (inference remains 100% local on device). To get one: Register for a free account on the [Moondream Console](https://moondream.ai) and copy your free key from the dashboard.
    * `OBSIDIAN_VAULT_PATH` – absolute path to the folder where `LiveContext.md` and `PersistentContext.md` are stored.
5. **Model files** – YOLO‑World CoreML package and MediaPipe hand‑landmarker task will be downloaded automatically on first run (requires internet).
6. **License** – The code is released under **AGPL‑v3**. See `LICENSE` for full terms.
7. **Run the application** – launch the project with `python3 main.py`.

---

## Experimental Disclaimer

This repository is a **very experimental proof‑of‑concept** built by a single 16-year-old developer. It is **not yet production‑ready**, nor is it safe or ready for autonomous manual‑task guidance or safety‑critical applications. The architecture, performance characteristics, and hardware recommendations are subject to change as the project evolves.

---

## License and Credits

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**. See the [LICENSE](LICENSE) file for the full text.

The project utilizes the following third-party components:

*   **YOLO-World**: Released under the [GPL-3.0 License](https://github.com/AILab-CVC/YOLO-World).
*   **Moondream2**: Released under the [Apache 2.0 License](https://huggingface.co/vikhyatk/moondream2).
*   **MediaPipe**: Developed by Google, released under the [Apache 2.0 License](https://github.com/google-ai-edge/mediapipe).
*   **OpenCV**: Released under the [Apache 2.0 License](https://github.com/opencv/opencv).

---
