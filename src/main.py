import sys
import os
import time
import threading
import cv2
import logging

logging.disable(logging.CRITICAL)
os.environ["YOLO_VERBOSE"] = "False"

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from modules.vision                    import VisionSystem
from modules.face_recognition_module   import FaceRecognitionModule
from modules.animal_recognition_module import AnimalRecognitionModule
from modules.voice_interaction         import VoiceInteraction
from modules.database                  import DatabaseManager


# ═════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════

CAMERA_INDEX            = 0
YOLO_MODEL              = "yolov8n.pt"
YOLO_CONFIDENCE         = 0.5
ANIMAL_SNAPSHOT_DIR     = os.path.join("data", "animal_snapshots")
KNOWN_LABEL_DISPLAY_SEC = 3

HUD_COLOR_KNOWN          = (0, 255, 100)
HUD_COLOR_UNKNOWN        = (0, 100, 255)
HUD_COLOR_ANIMAL_KNOWN   = (0, 200, 255)
HUD_COLOR_ANIMAL_UNKNOWN = (0, 130, 255)
HUD_COLOR_TEXT           = (255, 255, 255)


# ─── CLEAN TERMINAL PRINT ────────────────────────────────────────
def log(msg):
    """
    Single print function for all terminal output.
    Only this function should print — everything else is silent.
    """
    print(f"  {msg}")


# ═════════════════════════════════════════════════════════════════
# AI VISION SYSTEM
# ═════════════════════════════════════════════════════════════════

class AIVisionSystem:

    def __init__(self):

        print("\n" + "═" * 45)
        print("   AI VISION SYSTEM")
        print("═" * 45)

        # ─── SILENCE STDOUT DURING MODULE INIT ───────────────────
        # Redirects all print() calls from third-party modules to
        # /dev/null during startup so only our log() calls appear.
        import io
        _silent = open(os.devnull, "w")
        _real_stdout = sys.stdout
        sys.stdout = _silent

        self.db           = DatabaseManager()
        self.face_module  = FaceRecognitionModule()
        self.face_module.load_known_faces(self.db.get_all_humans())
        self.animal_module = AnimalRecognitionModule()
        self.animal_module.load_known_animals(self.db.get_all_animals())
        self.voice        = VoiceInteraction()
        self.vision       = VisionSystem(
            camera_index = CAMERA_INDEX,
            model_size   = YOLO_MODEL,
            confidence   = YOLO_CONFIDENCE
        )

        # ─── RESTORE STDOUT ──────────────────────────────────────
        sys.stdout = _real_stdout
        _silent.close()

        os.makedirs(ANIMAL_SNAPSHOT_DIR, exist_ok=True)

        # ─── STATE ───────────────────────────────────────────────
        self._voice_active        = False
        self._voice_lock          = threading.Lock()
        self._last_known_detected = 0
        self._unknown_first_seen  = {}
        self.UNKNOWN_PATIENCE_SEC = 10
        self._hud_labels          = {}

        known_count  = len(self.face_module.known_faces)
        animal_count = len(self.animal_module.known_animals)

        print(f"\n  Known humans  : {known_count}")
        print(f"  Known animals : {animal_count}")
        print(f"\n  System ready — press Q to quit")
        print("═" * 45 + "\n")

    # ═══════════════════════════════════════════════════════════════
    # HUMAN HANDLER
    # ═══════════════════════════════════════════════════════════════

    def _handle_human(self, frame, box):

        embedding = self.face_module.get_embedding(frame, box)
        if embedding is None:
            return

        name    = self.face_module.identify(embedding)
        box_key = "human"

        if name != "unknown":

            # ─── KNOWN ───────────────────────────────────────────
            log(f"Detected: {name}")
            self._last_known_detected = time.time()
            self._unknown_first_seen.pop("human", None)

            self._hud_labels[box_key] = {
                "label"  : f" {name}",
                "color"  : HUD_COLOR_KNOWN,
                "box"    : box,
                "expires": time.time() + KNOWN_LABEL_DISPLAY_SEC
            }

            threading.Thread(
                target=self.db.update_human_encounter,
                args=(name,), daemon=True
            ).start()

        else:

            # ─── GUARD ───────────────────────────────────────────
            if time.time() - self._last_known_detected < 3.0:
                return

            if "human" not in self._unknown_first_seen:
                self._unknown_first_seen["human"] = time.time()
                log("Unknown person detected")

            time_watching = time.time() - self._unknown_first_seen["human"]

            self._hud_labels[box_key] = {
                "label"  : " Unknown",
                "color"  : HUD_COLOR_UNKNOWN,
                "box"    : box,
                "expires": time.time() + KNOWN_LABEL_DISPLAY_SEC
            }

            if time_watching >= self.UNKNOWN_PATIENCE_SEC and \
               not self._voice_active and \
               self.face_module.should_trigger_voice(box):

                self._unknown_first_seen.pop("human", None)
                threading.Thread(
                    target=self._run_human_registration,
                    args=(frame, box, embedding), daemon=True
                ).start()

    # ═══════════════════════════════════════════════════════════════
    # HUMAN VOICE REGISTRATION
    # ═══════════════════════════════════════════════════════════════

    def _run_human_registration(self, frame, box, embedding):

        with self._voice_lock:
            self._voice_active = True
            try:
                name = self.voice.ask_for_name()
                if name is None:
                    log("Registration cancelled — no name captured")
                    return

                log(f"Registering: {name}")
                image_path = self.face_module.save_snapshot(frame, box, name)
                self.db.register_human(name, embedding, image_path)
                self.face_module.add_known_face(name, embedding)
                log(f"Saved: {name}")

            except Exception as e:
                log(f"Registration error: {e}")
            finally:
                self._voice_active = False

    # ═══════════════════════════════════════════════════════════════
    # ANIMAL HANDLER
    # ═══════════════════════════════════════════════════════════════

    def _handle_animal(self, frame, box, yolo_label):

        embedding = self.animal_module.get_embedding(frame, box)
        if embedding is None:
            return

        name    = self.animal_module.identify(embedding, yolo_label)
        box_key = yolo_label

        if name != "unknown":

            # ─── KNOWN ───────────────────────────────────────────
            log(f"Detected: {name} ({yolo_label})")
            self._unknown_first_seen.pop(yolo_label, None)

            self._hud_labels[box_key] = {
                "label"  : f" {name} ({yolo_label})",
                "color"  : HUD_COLOR_ANIMAL_KNOWN,
                "box"    : box,
                "expires": time.time() + KNOWN_LABEL_DISPLAY_SEC
            }

            threading.Thread(
                target=self.db.update_animal_encounter,
                args=(name, yolo_label), daemon=True
            ).start()

        else:

            if yolo_label not in self._unknown_first_seen:
                self._unknown_first_seen[yolo_label] = time.time()
                log(f"Unknown {yolo_label} detected")

            time_watching = time.time() - self._unknown_first_seen[yolo_label]

            self._hud_labels[box_key] = {
                "label"  : f" Unknown {yolo_label}",
                "color"  : HUD_COLOR_ANIMAL_UNKNOWN,
                "box"    : box,
                "expires": time.time() + KNOWN_LABEL_DISPLAY_SEC
            }

            if time_watching >= self.UNKNOWN_PATIENCE_SEC and \
               not self._voice_active and \
               self.animal_module.should_trigger_voice(box):

                self._unknown_first_seen.pop(yolo_label, None)
                threading.Thread(
                    target=self._run_animal_registration,
                    args=(frame, box, yolo_label, embedding), daemon=True
                ).start()

    # ═══════════════════════════════════════════════════════════════
    # ANIMAL VOICE REGISTRATION
    # ═══════════════════════════════════════════════════════════════

    def _run_animal_registration(self, frame, box, yolo_label, embedding):

        with self._voice_lock:
            self._voice_active = True
            try:
                name = self.voice.ask_for_animal_name(yolo_label)
                if name is None:
                    log(f"Animal registration cancelled")
                    return

                log(f"Registering: {name} ({yolo_label})")
                image_path = self.animal_module.save_snapshot(
                    frame, box, name, yolo_label
                )
                self.db.register_animal(name, yolo_label, embedding, image_path)
                self.animal_module.add_known_animal(name, yolo_label, embedding)
                log(f"Saved: {name} ({yolo_label})")

            except Exception as e:
                log(f"Animal registration error: {e}")
            finally:
                self._voice_active = False

    # ═══════════════════════════════════════════════════════════════
    # HUD OVERLAY
    # ═══════════════════════════════════════════════════════════════

    def _draw_hud(self, frame):

        now     = time.time()
        expired = []

        for key, entry in self._hud_labels.items():

            if now > entry["expires"]:
                expired.append(key)
                continue

            x1, y1, x2, y2 = entry["box"]
            label  = entry["label"]
            color  = entry["color"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - 14),
                (x1 + tw + 10, y1),
                color, -1
            )
            cv2.putText(
                frame, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                HUD_COLOR_TEXT, 2, cv2.LINE_AA
            )

        for key in expired:
            del self._hud_labels[key]

        if self._voice_active:
            cv2.putText(
                frame, "Listening...",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2, cv2.LINE_AA
            )

        return frame

    # ═══════════════════════════════════════════════════════════════
    # RUN
    # ═══════════════════════════════════════════════════════════════

    def run(self):

        import cv2 as _cv2
        original_imshow = _cv2.imshow

        def patched_imshow(window_name, frame):
            self._draw_hud(frame)
            original_imshow(window_name, frame)

        _cv2.imshow = patched_imshow

        try:
            self.vision.run(
                on_human  = self._handle_human,
                on_animal = self._handle_animal,
            )
        finally:
            _cv2.imshow = original_imshow
            self.db.close()
            print("\n  System shut down.")
            print("═" * 45 + "\n")


# ═════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    system = AIVisionSystem()
    system.run()