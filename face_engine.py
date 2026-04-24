import face_recognition
import os
import numpy as np
import threading
from PIL import Image

# ── Constants ────────────────────────────────────────────────
KNOWN_FACES_DIR   = "data/known_faces"

# Supported image formats only — prevents loading .txt, .DS_Store, etc.
VALID_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

#  Fix: use face_distance threshold instead of a boolean match
# Lower = stricter. 0.5 is a safe default; lower to 0.45 for tighter security.
MATCH_TOLERANCE   = 0.5

# ── Thread-safe global state ─────────────────────────────────
_lock             = threading.Lock()
known_encodings: list[np.ndarray] = []
known_names:     list[str]        = []


# ── Load known faces ─────────────────────────────────────────
def load_known_faces(path: str = KNOWN_FACES_DIR) -> None:
    """
    Scans the given directory and builds the known-face database.
    - Skips non-image files silently.
    - Skips unreadable / face-less images with a warning.
    - Thread-safe: replaces global state atomically.
    - Safe to call multiple times (re-loads cleanly).
    """
    #  Fix: check directory exists before listing
    if not os.path.isdir(path):
        print(f"[face_engine] WARNING: Known faces directory not found: '{path}'")
        return

    new_encodings: list[np.ndarray] = []
    new_names:     list[str]        = []

    for file in os.listdir(path):
        ext = os.path.splitext(file)[1].lower()

        #  Fix: skip non-image files
        if ext not in VALID_EXTENSIONS:
            continue

        #  Fix: use os.path.join for cross-platform compatibility
        filepath = os.path.join(path, file)

        try:
            pil_img=Image.open(filepath).convert("RGB")
            img = np.array(pil_img)
            
            encodings = face_recognition.face_encodings(img)

            if not encodings:
                print(f"[face_engine] WARNING: No face found in '{file}' — skipping.")
                continue

            if len(encodings) > 1:
                print(f"[face_engine] WARNING: Multiple faces in '{file}' — using first face only.")

            new_encodings.append(encodings[0])
            #  Fix: use os.path.splitext consistently for the name
            new_names.append(os.path.splitext(file)[0])

        except Exception as e:
            #  Fix: one bad image doesn't crash the whole load
            print(f"[face_engine] ERROR loading '{file}': {e} — skipping.")

    #  Fix: replace global state atomically under lock
    with _lock:
        known_encodings.clear()
        known_names.clear()
        known_encodings.extend(new_encodings)
        known_names.extend(new_names)

    print(f"[face_engine] Loaded {len(known_encodings)} known face(s) from '{path}'.")


# ── Recognise faces in a frame ───────────────────────────────
def recognize_face(frame: np.ndarray) -> tuple[list, list[str]]:
    """
    Detects and identifies faces in a BGR frame.

    Args:
        frame: A BGR numpy array (as returned by OpenCV).

    Returns:
        (face_locations, names)
        face_locations — list of (top, right, bottom, left) tuples
        names          — parallel list of name strings or "Unknown"
    """
    #  Fix: validate the frame before processing
    if (frame is None or
            not isinstance(frame, np.ndarray) or
            frame.ndim != 3 or
            frame.shape[2] != 3):
        print("[face_engine] WARNING: Invalid frame received — skipping recognition.")
        return [], []

    #  Fix: guard against empty known-face database
    with _lock:
        if not known_encodings:
            return [], []
        enc_snapshot  = list(known_encodings)
        name_snapshot = list(known_names)

    # face_recognition expects RGB; OpenCV gives BGR → flip channels
    rgb_frame  = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    names: list[str] = []

    for face_enc in face_encodings:
        #  Fix: use face_distance to find the BEST (closest) match
        #         instead of matches.index(True) which picks the FIRST match
        distances = face_recognition.face_distance(enc_snapshot, face_enc)
        best_idx  = int(np.argmin(distances))
        best_dist = distances[best_idx]

        #  Fix: only accept if distance is within tolerance threshold
        if best_dist <= MATCH_TOLERANCE:
            name = name_snapshot[best_idx]
        else:
            name = "Unknown"

        names.append(name)

    return face_locations, names