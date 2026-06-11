# AI Vision System

A real-time AI-powered vision system that uses a webcam to perform face recognition, animal detection, and voice-driven registration — all running simultaneously via a multi-threaded pipeline.

## Features

- **Face Recognition** — Identifies registered people by name; labels unrecognized visitors as "Unknown"
- **Animal Detection** — Detects 10 animal species using YOLOv8 and saves snapshots automatically
- **Voice Registration** — After 10 seconds of watching an unknown face or animal, the system speaks and asks for a name via microphone, then registers and remembers them
- **HUD Overlay** — Color-coded bounding boxes and name labels drawn live on the camera feed
- **Local JSON Storage** — All registered humans, named animals, and encounter logs are persisted to `data/local_storage.json`

## HUD Color Guide

| Color | Meaning |
|---|---|
| Green `(0, 255, 100)` | Known human |
| Orange `(0, 100, 255)` | Unknown human |
| Cyan `(0, 200, 255)` | Known animal |
| Blue `(0, 130, 255)` | Unknown animal |
| Yellow `Listening...` | Voice registration in progress |

## Project Structure

```text
vision_system/
├── .env                          # Optional: override storage path
├── yolov8n.pt                    # YOLOv8 nano weights
├── data/
│   ├── local_storage.json        # Registered humans, animals, encounter logs
│   ├── known_faces/              # Auto-saved face snapshots on registration
│   └── animal_snapshots/         # Auto-saved animal snapshots on registration
└── src/
    ├── main.py                   # Entry point and main orchestration loop
    └── modules/
        ├── vision.py             # YOLOv8 detection loop + camera handling
        ├── face_recognition_module.py
        ├── animal_recognition_module.py
        ├── voice_interaction.py  # TTS + STT via Windows SAPI / SpeechRecognition
        └── database.py           # Local JSON storage manager
```

## Requirements

- Python 3.10+
- Windows (voice interaction uses Windows SAPI)
- Webcam
- CUDA-capable GPU recommended (runs on CPU but slower)

## Setup

1. **Clone / download** the project.

2. **Create and activate the virtual environment:**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install opencv-python ultralytics face-recognition speechrecognition pyaudio torch torchvision python-dotenv
   ```

4. **(Optional) Configure storage path** — by default data is saved to `data/local_storage.json`. To override, create a `.env` file at the project root:
   ```env
   JSON_STORAGE_FILE=data/local_storage.json
   ```

5. **YOLOv8 model** — `yolov8n.pt` is already included. If missing, the app will auto-download it on first run.

## Running

```powershell
cd src
python main.py
```

Press **Q** in the video window to quit.

## How Registration Works

1. An unknown face or animal appears on camera.
2. After **10 seconds** of continuous detection, the system triggers voice registration.
3. The system speaks (TTS) and asks for a name.
4. The user says the name into the microphone (STT).
5. A snapshot is saved and the person/animal is registered to local storage.
6. From that point on, they are recognized by name.

> Only one voice registration can run at a time. The "Listening..." overlay appears on screen during this process.

## Detected Animal Classes

`bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`

## Configuration

Key constants at the top of [src/main.py](src/main.py):

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index |
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model weights file |
| `YOLO_CONFIDENCE` | `0.5` | Minimum detection confidence (0–1) |
| `ANIMAL_SNAPSHOT_DIR` | `data/animal_snapshots` | Where animal snapshots are saved |
| `KNOWN_LABEL_DISPLAY_SEC` | `3` | Seconds to display a name label after detection |
| `UNKNOWN_PATIENCE_SEC` | `10` | Seconds before voice registration triggers for unknowns |
