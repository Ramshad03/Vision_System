# ─────────────────────────────────────────────────────────────────
# MODULE: face_recognition_module.py
# ROLE  : Extracts facial embeddings from camera frames,
#         compares them against known faces stored in memory,
#         and determines if a detected person is known or unknown.
#         Does NOT handle database reads/writes directly —
#         that is handled by database.py
# ─────────────────────────────────────────────────────────────────

import face_recognition
import numpy as np
import cv2
import os
import time


# ─── CONFIGURATION ───────────────────────────────────────────────
MATCH_TOLERANCE      = 0.38     # Lower = stricter matching (0.4–0.6 recommended)
FACE_PADDING         = 20      # Pixels to pad around face before saving snapshot
SNAPSHOT_DIR         = os.path.join("data", "known_faces")
UNKNOWN_COOLDOWN_SEC = 10      # Seconds before re-triggering voice for same unknown face


# ─── FACE RECOGNITION MODULE CLASS ───────────────────────────────
class FaceRecognitionModule:
    """
    Handles all face recognition logic.

    Flow:
      1. Receive full frame + YOLO box from the vision module.
      2. Verify a face is actually visible in the crop.
      3. Extract 128-dimension embedding using the box as face location.
      4. Compare embedding against known_faces list.
      5. Return recognition result: known name or 'unknown'
    """

    def __init__(self):

        # ─── KNOWN FACES STORE ───────────────────────────────────
        self.known_faces = []

        # ─── UNKNOWN FACE COOLDOWN TRACKER ───────────────────────
        self._unknown_cooldown = {}

        # ─── OPENCV HAAR CASCADE (face presence check) ────────────
        # Used instead of dlib HOG to avoid Windows dlib format errors.
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

        # ─── ENSURE SNAPSHOT DIRECTORY EXISTS ────────────────────
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        print("[FaceRecognition] Module initialised.")

    # ─── LOAD KNOWN FACES ────────────────────────────────────────
    def load_known_faces(self, records):
        """
        Populates the known_faces list from database records.
        Called at startup by main.py after fetching from MongoDB.

        Args:
            records : list of dicts with keys 'name' and 'embedding'
        """
        self.known_faces = [
            {"name": r["name"], "embedding": np.array(r["embedding"])}
            for r in records
        ]
        print(f"[FaceRecognition] Loaded {len(self.known_faces)} known face(s) from database.")

    # ─── ADD NEW FACE TO MEMORY ──────────────────────────────────
    def add_known_face(self, name, embedding):
        """
        Adds a newly registered person to the in-memory known_faces list.
        Called after successful voice registration so the system
        recognises them immediately without restarting.

        Args:
            name      : str      — confirmed name of the person
            embedding : np.array — 128-float facial embedding
        """
        self.known_faces.append({"name": name, "embedding": np.array(embedding)})
        print(f"[FaceRecognition] '{name}' added to in-memory known faces.")

    # ─── EXTRACT EMBEDDING FROM FRAME ────────────────────────────
    def get_embedding(self, frame, box):
        """
        Extracts a 128-dimension facial embedding ONLY if a face is
        actually visible inside the YOLO bounding box.

        First runs face detection on the cropped region — if no face
        is found (person viewed from behind, face turned away, too far)
        returns None immediately and skips identification entirely.

        This prevents garbage embeddings from body crops accidentally
        matching stored faces when no face is actually visible.

        Args:
            frame : ndarray — full BGR camera frame from OpenCV
            box   : tuple   — (x1, y1, x2, y2) bounding box from YOLO

        Returns:
            embedding : np.array of 128 floats, or None if no face found
        """

        # ─── NORMALISE FRAME TO CONTIGUOUS BGR UINT8 ─────────────
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = np.ascontiguousarray(frame)

        # ─── CLAMP BOX TO FRAME BOUNDARIES ───────────────────────
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # ─── STEP 1: VERIFY FACE IN CROP (OpenCV Haar, no dlib) ──
        # dlib HOG throws RuntimeError on some Windows builds for any
        # image format. OpenCV Haar cascade is format-agnostic and fast.
        bgr_crop = np.ascontiguousarray(frame[y1:y2, x1:x2], dtype=np.uint8)

        if bgr_crop.size == 0 or bgr_crop.shape[0] < 20 or bgr_crop.shape[1] < 20:
            return None

        gray_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        faces_found = self._face_cascade.detectMultiScale(
            gray_crop, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        if not len(faces_found):
            return None

        # ─── STEP 2: EXTRACT EMBEDDING FROM FULL FRAME ───────────
        # Convert YOLO box to dlib format: (top, right, bottom, left)
        # YOLO gives: x1=left, y1=top, x2=right, y2=bottom
        face_location = [(y1, x2, y2, x1)]
        rgb_frame = np.ascontiguousarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )

        try:
            encodings = face_recognition.face_encodings(
                rgb_frame,
                known_face_locations=face_location,
                num_jitters=1,
                model="small"
            )
        except Exception as e:
            print(f"[FaceRecognition] Embedding error: {e}")
            return None

        if not encodings:
            return None

        return encodings[0]

    # ─── COMPARE AGAINST KNOWN FACES ─────────────────────────────
    def identify(self, embedding):
        """
        Compares an embedding against all known faces.

        Args:
            embedding : np.array — 128-float embedding to identify

        Returns:
            name : str — matched name if known, or 'unknown'
        """

        if not self.known_faces or embedding is None:
            return "unknown"

        # ─── COMPUTE DISTANCES ───────────────────────────────────
        known_embeddings = [f["embedding"] for f in self.known_faces]
        distances = face_recognition.face_distance(known_embeddings, embedding)

        # ─── FIND BEST MATCH ─────────────────────────────────────
        best_index    = int(np.argmin(distances))
        best_distance = distances[best_index]

        if best_distance <= MATCH_TOLERANCE:
            matched_name = self.known_faces[best_index]["name"]
            return matched_name

        return "unknown"

    # ─── CHECK UNKNOWN COOLDOWN ───────────────────────────────────
    def should_trigger_voice(self, box):
        """
        Prevents voice interaction from firing repeatedly for the
        same unrecognised face in quick succession.

        Args:
            box : tuple — (x1, y1, x2, y2)

        Returns:
            bool — True if voice interaction should be triggered
        """
        x1, y1, x2, y2 = box
        key = (x1 // 50, y1 // 50, x2 // 50, y2 // 50)

        now          = time.time()
        last_trigger = self._unknown_cooldown.get(key, 0)

        if now - last_trigger >= UNKNOWN_COOLDOWN_SEC:
            self._unknown_cooldown[key] = now
            return True

        return False

    # ─── SAVE FACE SNAPSHOT ──────────────────────────────────────
    def save_snapshot(self, frame, box, name):
        """
        Saves a cropped face image to data/known_faces/ after registration.

        Args:
            frame : ndarray — full camera frame
            box   : tuple   — (x1, y1, x2, y2)
            name  : str     — confirmed name used as the filename

        Returns:
            image_path : str — path to the saved image file
        """
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1p = max(0, x1 - FACE_PADDING)
        y1p = max(0, y1 - FACE_PADDING)
        x2p = min(w, x2 + FACE_PADDING)
        y2p = min(h, y2 + FACE_PADDING)
        face_crop = frame[y1p:y2p, x1p:x2p]

        filename   = f"{name.replace(' ', '_')}_{int(time.time())}.jpg"
        image_path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(image_path, face_crop)

        print(f"[FaceRecognition] Snapshot saved: {image_path}")
        return image_path