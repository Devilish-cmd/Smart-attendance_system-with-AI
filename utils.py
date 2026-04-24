import pandas as pd
import os
import threading
from datetime import datetime

# ── Constants ───────────────────────────────────────────────
ATTENDANCE_FILE = "data/attendance.csv"

# On-time cutoff — entry AT or before this time = Present, after = Late
CUTOFF_HOUR   = 10
CUTOFF_MINUTE = 0

# Required columns (must match chatbot.py expectations)
COLUMNS = ["Name", "Date", "Time", "Status"]

# Thread lock — prevents CSV corruption when multiple faces are detected at once
_lock = threading.Lock()


# ── Helper: get or create DataFrame ────────────────────────
def _load_df() -> pd.DataFrame:
    """Load CSV if it exists and is valid, otherwise return empty DataFrame."""
    if not os.path.exists(ATTENDANCE_FILE):
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=COLUMNS)

    # Add any missing columns gracefully
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df


# ── Helper: ensure output directory exists ─────────────────
def _ensure_dir():
    os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)


# ── Main function ───────────────────────────────────────────
def mark_attendance(name: str) -> str:
    """
    Marks attendance for a given name.

    Returns one of:
      - "Attendance Marked - Present"
      - "Attendance Marked - Late"
      - "Already Marked"
      - "Error: ..." on unexpected failure
    """
    if not name or not name.strip():
        return "Error: Invalid name provided."

    name = name.strip()
    now  = datetime.now()

    today        = now.strftime("%Y-%m-%d")   # e.g. "2024-07-15"
    current_time = now.strftime("%H:%M")       # e.g. "09:45"

    #  Fix: compare time objects, NOT strings
    cutoff  = now.replace(hour=CUTOFF_HOUR, minute=CUTOFF_MINUTE, second=0, microsecond=0)
    is_late = now > cutoff

    with _lock:   # Fix: thread-safe file access
        try:
            _ensure_dir()           # ✅ Fix: create data/ if missing
            df = _load_df()

            #  Fix: check for TODAY's record only (not all-time)
            already_marked = (
                (df["Name"].str.strip().str.lower() == name.lower()) &
                (df["Date"].astype(str).str.strip() == today)
            ).any()

            if already_marked:
                return "Already Marked"

            #  Fix: record Status as Present or Late (not just block late entry)
            status = "Late" if is_late else "Present"

            #  Fix: use pd.concat instead of df.loc[len(df)] to avoid index gaps
            new_row = pd.DataFrame([[name, today, current_time, status]], columns=COLUMNS)
            df = pd.concat([df, new_row], ignore_index=True)

            df.to_csv(ATTENDANCE_FILE, index=False)
            return f"Attendance Marked - {status}"

        except OSError as e:
            return f"Error: Could not write attendance file — {e}"
        except Exception as e:
            return f"Error: Unexpected issue — {e}"