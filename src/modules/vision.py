import cv2
from ultralytics import YOLO

# ─── YOLO ANIMAL CLASSES ─────────────────────────────────────────
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

# ─── HUD COLORS (BGR) ────────────────────────────────────────────
COLOR_ANIMAL = (0, 165, 255)
COLOR_TEXT   = (255, 255, 255)


class VisionSystem:

    def __init__(self, camera_index=0, model_size="yolov8n.pt", confidence=0.5):
        print("[Vision] Loading YOLOv8 model...")
        self.model      = YOLO(model_size)
        self.confidence = confidence
        print(f"[Vision] Model loaded: {model_size}")

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"[Vision] ERROR: Could not open camera at index {camera_index}.")
        print(f"[Vision] Camera opened at index {camera_index}.")

    # ─── KEEP BEST DETECTION PER CLASS ───────────────────────────
    def _keep_best_per_class(self, detections):
        """
        From all raw detections, keeps only the single highest-confidence
        detection per class label per frame.

        This is the simplest and most reliable way to prevent multiple
        boxes on the same person or animal. YOLO often detects the same
        entity as 2-5 overlapping instances with different confidence
        scores — we simply keep the winner.

        Args:
            detections : list of dicts — label, confidence, box

        Returns:
            list of dicts — at most one entry per class label
        """
        best = {}
        for det in detections:
            label = det["label"]
            if label not in best or det["confidence"] > best[label]["confidence"]:
                best[label] = det
        return list(best.values())

    # ─── ROUTE DETECTION ─────────────────────────────────────────
    def _route_detection(self, label, frame, box, on_human, on_animal):
        """
        Routes detection to correct pipeline.
        Human boxes are drawn by main.py HUD — not here.
        Animal boxes are drawn here.
        """
        if label == "person":
            on_human(frame, box)
            return None, None                       # HUD draws human boxes

        elif label in ANIMAL_CLASSES:
            on_animal(frame, box, label)
            return COLOR_ANIMAL, f"Animal: {label}"

        return None, None

    # ─── DRAW BOX ────────────────────────────────────────────────
    def _draw_box(self, frame, box, label, color, confidence):
        x1, y1, x2, y2 = box
        shrink = 10
        cv2.rectangle(frame, (x1 + shrink, y1 + shrink), (x2 - shrink, y2 - shrink), color, 2)
        text = f"{label} {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    # ─── MAIN LOOP ───────────────────────────────────────────────
    def run(self, on_human, on_animal):
        print("[Vision] Starting detection loop. Press Q to quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[Vision] ERROR: Lost camera feed.")
                break

            # ─── YOLO INFERENCE ──────────────────────────────────
            results = self.model(frame, conf=self.confidence, verbose=False)

            # ─── COLLECT HUMAN & ANIMAL DETECTIONS ───────────────
            raw = []
            for result in results:
                for box_data in result.boxes:
                    label = self.model.names[int(box_data.cls[0])]
                    if label != "person" and label not in ANIMAL_CLASSES:
                        continue
                    x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                    raw.append({
                        "label"     : label,
                        "confidence": float(box_data.conf[0]),
                        "box"       : (x1, y1, x2, y2)
                    })

            # ─── KEEP SINGLE BEST PER CLASS ──────────────────────
            # One person box maximum + one box per animal species
            best = self._keep_best_per_class(raw)

            # ─── PROCESS & DRAW ───────────────────────────────────
            for det in best:
                color, display_label = self._route_detection(
                    det["label"], frame, det["box"], on_human, on_animal
                )
                if color is not None:
                    self._draw_box(frame, det["box"], display_label, color, det["confidence"])

            cv2.imshow("AI Vision System — Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Vision] Quit signal received.")
                break

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("[Vision] Camera released.")