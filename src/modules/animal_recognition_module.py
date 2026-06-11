import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
import time


# ─── CONFIGURATION ───────────────────────────────────────────────
MATCH_THRESHOLD      = 0.50    # Cosine similarity threshold (0.0–1.0)
                                # Lowered from 0.75 — MobileNet embeddings vary
                                # significantly across frames for the same animal.
                                # Raise if you get false positives between animals.
SNAPSHOT_DIR         = os.path.join("data", "animal_snapshots")
UNKNOWN_COOLDOWN_SEC = 15      # Seconds before re-triggering voice for same animalide


class AnimalRecognitionModule:
    """
    Visual similarity-based animal recognition.

    Uses MobileNetV2 as a feature extractor — removes the final
    classification layer and uses the 1280-dim feature vector as
    an embedding fingerprint for each individual animal.

    Two cats of different colours produce very different embeddings,
    so the same cat will consistently match itself while a new cat
    triggers the registration dialogue.
    """

    def __init__(self):

        # ─── LOAD PRETRAINED MOBILENET ────────────────────────────
        print("[AnimalRecognition] Loading MobileNetV2 feature extractor...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MobileNetV2 pretrained on ImageNet
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Remove the classifier — keep only the feature extractor
        # Output: 1280-dimensional embedding vector per image
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        print(f"[AnimalRecognition] Model ready on {self.device}.")

        # ─── IMAGE PREPROCESSING ──────────────────────────────────
        # MobileNetV2 expects 224x224 RGB images normalised to ImageNet stats
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ─── KNOWN ANIMALS STORE ──────────────────────────────────
        # Format: [{"name": str, "label": str, "embedding": np.array}, ...]
        self.known_animals = []

        # ─── COOLDOWN TRACKER ─────────────────────────────────────
        self._cooldown = {}

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        print("[AnimalRecognition] Module initialised.")

    # ─── LOAD KNOWN ANIMALS FROM DB ──────────────────────────────
    def load_known_animals(self, records):
        """
        Populates known_animals from database records at startup.

        Args:
            records : list of dicts with keys: name, label, embedding
        """
        self.known_animals = [
            {
                "name"     : r["name"],
                "label"    : r["label"],
                "embedding": np.array(r["embedding"])
            }
            for r in records
        ]
        print(f"[AnimalRecognition] Loaded {len(self.known_animals)} known animal(s).")

    # ─── ADD TO IN-MEMORY STORE ───────────────────────────────────
    def add_known_animal(self, name, label, embedding):
        """
        Adds a newly registered animal to in-memory store immediately.

        Args:
            name      : str      — confirmed animal name
            label     : str      — YOLO class (e.g. 'cat', 'dog')
            embedding : np.array — 1280-dim feature vector
        """
        self.known_animals.append({
            "name"     : name,
            "label"    : label,
            "embedding": np.array(embedding)
        })
        print(f"[AnimalRecognition] '{name}' ({label}) added to in-memory store.")

    # ─── EXTRACT EMBEDDING ────────────────────────────────────────
    def get_embedding(self, frame, box):
        """
        Extracts a 1280-dim visual embedding from the animal crop.

        Args:
            frame : ndarray — full BGR camera frame
            box   : tuple   — (x1, y1, x2, y2) YOLO bounding box

        Returns:
            embedding : np.array of 1280 floats, or None on failure
        """
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]

        # ─── CROP ANIMAL REGION ───────────────────────────────────
        pad = 10
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(w, x2 + pad)
        y2c = min(h, y2 + pad)
        crop = frame[y1c:y2c, x1c:x2c]

        if crop.size == 0:
            return None

        # ─── CONVERT BGR TO RGB ───────────────────────────────────
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        try:
            # ─── PREPROCESS & EXTRACT EMBEDDING ──────────────────
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(tensor)

            # Flatten to 1D vector and convert to numpy
            embedding = features.squeeze().cpu().numpy().flatten()
            return embedding

        except Exception as e:
            print(f"[AnimalRecognition] Embedding error: {e}")
            return None

    # ─── IDENTIFY ANIMAL ─────────────────────────────────────────
    def identify(self, embedding, yolo_label):
        """
        Compares embedding against known animals of the same species
        using cosine similarity.

        Only compares within the same YOLO class — a dog embedding
        is never compared against cat embeddings.

        Args:
            embedding  : np.array — 1280-dim feature vector
            yolo_label : str      — YOLO class label for filtering

        Returns:
            name : str — matched animal name, or 'unknown'
        """
        if embedding is None:
            return "unknown"

        # ─── FILTER TO SAME SPECIES ONLY ──────────────────────────
        same_species = [
            a for a in self.known_animals
            if a["label"] == yolo_label
        ]

        if not same_species:
            return "unknown"

        # ─── NORMALISE QUERY EMBEDDING ONCE ──────────────────────
        norm_a = np.linalg.norm(embedding)
        if norm_a == 0:
            return "unknown"
        unit_a = embedding / norm_a

        # ─── COSINE SIMILARITY AGAINST ALL KNOWN ─────────────────
        best_name  = "unknown"
        best_score = -1.0

        for animal in same_species:
            known_emb = animal["embedding"]

            norm_b = np.linalg.norm(known_emb)
            if norm_b == 0:
                continue

            unit_b = known_emb / norm_b
            score  = float(np.dot(unit_a, unit_b))

            # ─── DEBUG: show score for every candidate ────────────
            # Remove this line once recognition is working reliably
            print(f"  [AnimalRecog] '{animal['name']}' score: {score:.4f} (threshold: {MATCH_THRESHOLD})")

            if score > best_score:
                best_score = score
                if score >= MATCH_THRESHOLD:
                    best_name = animal["name"]

        if best_name != "unknown":
            print(f"  [AnimalRecog] Matched: '{best_name}' ({best_score:.4f})")
        else:
            print(f"  [AnimalRecog] No match for '{yolo_label}' — best score: {best_score:.4f}")

        return best_name

    # ─── COOLDOWN CHECK ───────────────────────────────────────────
    def should_trigger_voice(self, box):
        """
        Prevents voice dialogue from firing every frame for the same animal.

        Args:
            box : tuple — (x1, y1, x2, y2)

        Returns:
            bool — True if enough time has passed
        """
        x1, y1, x2, y2 = box
        key = (x1 // 80, y1 // 80)
        now = time.time()

        if now - self._cooldown.get(key, 0) >= UNKNOWN_COOLDOWN_SEC:
            self._cooldown[key] = now
            return True
        return False

    # ─── SAVE SNAPSHOT ───────────────────────────────────────────
    def save_snapshot(self, frame, box, name, label):
        """
        Saves a cropped animal image for database storage.

        Args:
            frame : ndarray — full camera frame
            box   : tuple   — (x1, y1, x2, y2)
            name  : str     — confirmed animal name
            label : str     — YOLO class label

        Returns:
            image_path : str
        """
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        pad  = 15
        crop = frame[
            max(0, y1 - pad) : min(h, y2 + pad),
            max(0, x1 - pad) : min(w, x2 + pad)
        ]

        filename   = f"{label}_{name.replace(' ', '_')}_{int(time.time())}.jpg"
        image_path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(image_path, crop)

        print(f"[AnimalRecognition] Snapshot saved: {image_path}")
        return image_path