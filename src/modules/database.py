import base64
import json
import os
import tempfile
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_FILE = PROJECT_ROOT / "data" / "local_storage.json"
_storage_file_setting = Path(os.getenv("JSON_STORAGE_FILE", str(DEFAULT_STORAGE_FILE)))
STORAGE_FILE = (
    _storage_file_setting
    if _storage_file_setting.is_absolute()
    else PROJECT_ROOT / _storage_file_setting
).resolve()


class DatabaseManager:
    """
    Local JSON-backed storage for registered humans and animals.

    The public methods intentionally match the old database manager so the
    vision, face-recognition, and animal-recognition modules can keep using the
    same calls without knowing where the data is stored.
    """

    def __init__(self, storage_file=None):
        self.storage_file = Path(storage_file or STORAGE_FILE)
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_storage_file()
        print(f"[Storage] Using local JSON file: {self.storage_file}")

    def _ensure_storage_file(self):
        if not self.storage_file.exists():
            self._write_data({"humans": [], "animals": []})
            return

        try:
            data = self._read_data()
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[Storage] Could not read JSON storage file: {self.storage_file}\n"
                f"Fix or delete the file and restart.\nError: {exc}"
            )

        changed = False
        for key in ("humans", "animals"):
            if key not in data or not isinstance(data[key], list):
                data[key] = []
                changed = True

        if changed:
            self._write_data(data)

    def _read_data(self):
        with self.storage_file.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write_data(self, data):
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=f"{self.storage_file.stem}_",
            suffix=".tmp",
            dir=self.storage_file.parent,
            text=True,
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
                file.write("\n")
            os.replace(temp_path, self.storage_file)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _load(self):
        with self._lock:
            return self._read_data()

    def _save(self, data):
        with self._lock:
            self._write_data(data)

    def _new_id(self):
        return uuid.uuid4().hex

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def _encode_image(self, image_path):
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")
        return None

    def _embedding_to_list(self, embedding):
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding

    def _without_image_b64(self, records):
        cleaned = []
        for record in records:
            item = deepcopy(record)
            item.pop("image_b64", None)
            cleaned.append(item)
        return cleaned

    # HUMAN OPERATIONS

    def get_all_humans(self):
        data = self._load()
        records = self._without_image_b64(data.get("humans", []))
        print(f"[Storage] Loaded {len(records)} human record(s).")
        return records

    def register_human(self, name, embedding, image_path):
        now = self._now()
        document = {
            "id": self._new_id(),
            "name": name,
            "embedding": self._embedding_to_list(embedding),
            "image_path": image_path,
            "image_b64": self._encode_image(image_path),
            "first_seen": now,
            "last_seen": now,
            "encounters": 1,
        }

        with self._lock:
            data = self._read_data()
            data.setdefault("humans", []).append(document)
            self._write_data(data)

        print(f"[Storage] Human registered: '{name}' (ID: {document['id']})")
        return document["id"]

    def update_human_encounter(self, name):
        matched = False

        with self._lock:
            data = self._read_data()
            for human in data.get("humans", []):
                if human.get("name") == name:
                    human["last_seen"] = self._now()
                    human["encounters"] = int(human.get("encounters", 0)) + 1
                    matched = True
                    break

            if matched:
                self._write_data(data)

        if matched:
            print(f"[Storage] Encounter updated for '{name}'.")
        else:
            print(f"[Storage] WARNING: No record found for '{name}' to update.")

    def human_exists(self, name):
        data = self._load()
        return any(human.get("name") == name for human in data.get("humans", []))

    # ANIMAL OPERATIONS

    def register_animal(self, name, label, embedding, image_path):
        now = self._now()
        document = {
            "id": self._new_id(),
            "name": name,
            "label": label,
            "category": "animal",
            "embedding": self._embedding_to_list(embedding),
            "image_path": image_path,
            "image_b64": self._encode_image(image_path),
            "first_seen": now,
            "last_seen": now,
            "encounters": 1,
        }

        with self._lock:
            data = self._read_data()
            data.setdefault("animals", []).append(document)
            self._write_data(data)

        print(f"[Storage] Animal registered: '{name}' ({label}) ID: {document['id']}")
        return document["id"]

    def get_all_animals(self):
        data = self._load()
        records = [
            record for record in data.get("animals", [])
            if record.get("name") and record.get("label") and record.get("embedding")
        ]
        records = self._without_image_b64(records)
        print(f"[Storage] Loaded {len(records)} named animal record(s).")

        for record in records:
            emb_len = len(record.get("embedding", []))
            print(f"  - '{record['name']}' ({record['label']}) embedding length: {emb_len}")

        return records

    def update_animal_encounter(self, name, label):
        matched = False

        with self._lock:
            data = self._read_data()
            for animal in data.get("animals", []):
                if animal.get("name") == name and animal.get("label") == label:
                    animal["last_seen"] = self._now()
                    animal["encounters"] = int(animal.get("encounters", 0)) + 1
                    matched = True
                    break

            if matched:
                self._write_data(data)

        if matched:
            print(f"[Storage] Encounter updated for animal '{name}' ({label}).")
        else:
            print(f"[Storage] WARNING: No animal record found for '{name}' ({label}).")

    def log_animal(self, label, confidence, image_path):
        document = {
            "id": self._new_id(),
            "category": "animal_detection_log",
            "yolo_label": label,
            "confidence": round(confidence, 4),
            "image_path": image_path,
            "image_b64": self._encode_image(image_path),
            "timestamp": self._now(),
        }

        with self._lock:
            data = self._read_data()
            data.setdefault("animals", []).append(document)
            self._write_data(data)

        print(f"[Storage] Animal detection logged: '{label}' ({confidence:.0%})")
        return document["id"]

    def get_recent_animals(self, limit=10):
        data = self._load()
        records = self._without_image_b64(data.get("animals", []))
        records.sort(
            key=lambda item: item.get("timestamp") or item.get("last_seen") or "",
            reverse=True,
        )
        return records[:limit]

    # CONNECTION MANAGEMENT

    def close(self):
        print("[Storage] Local JSON storage closed.")
