import pandas as pd
import os
from datetime import date


# ── Constants ──────────────────────────────────────────────
CSV_PATH        = "data/attendance.csv"
REQUIRED_COLS   = {"Name", "Date", "Status"}   # expected column names


# ── Helper: load & validate CSV ────────────────────────────
def load_attendance() -> tuple[pd.DataFrame | None, str | None]:
    """
    Returns (dataframe, None) on success,
    or (None, error_message) on failure.
    """
    if not os.path.exists(CSV_PATH):
        return None, "Attendance file not found. No records available yet."

    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        return None, "Attendance file is empty. No records available yet."
    except pd.errors.ParserError:
        return None, "Attendance file is corrupted or in an unexpected format."
    except OSError as e:
        return None, f"Could not read attendance file: {e}"

    # Validate required columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return None, f"Attendance file is missing columns: {', '.join(missing)}"

    # Normalise name column
    df["Name"] = df["Name"].astype(str).str.strip()

    return df, None


# ── Helper: extract a person's name from a query ───────────
def extract_name(query: str, df: pd.DataFrame) -> str | None:
    """
    Checks if any known name (case-insensitive) appears in the query.
    Returns the matched name as stored in the DataFrame, or None.
    """
    for name in df["Name"].unique():
        if name.lower() in query:
            return name
    return None


# ── Main chatbot function ──────────────────────────────────
def chatbot_response(query: str) -> str:
    if not query or not query.strip():
        return "Please type a question."

    query = query.lower().strip()
    today = str(date.today())          # e.g. "2024-07-15"

    df, error = load_attendance()
    if error:
        return error

    # Filter for today's records
    today_df = df[df["Date"].astype(str).str.strip() == today]

    # ── "Who is present today?" ────────────────────────────
    if any(phrase in query for phrase in ["who is present", "who attended", "list present"]):
        if today_df.empty:
            return "No attendance has been recorded for today yet."
        names = today_df["Name"].drop_duplicates().tolist()
        return f"Present today ({today}): {', '.join(names)}"

    # ── "How many are present today?" ─────────────────────
    if any(phrase in query for phrase in ["how many present", "total present", "count present"]):
        count = today_df["Name"].nunique()
        return f"{count} student(s) are present today ({today})."

    # ── "Is [Name] present?" ───────────────────────────────
    if any(phrase in query for phrase in ["is", "present", "attended", "marked"]):
        name = extract_name(query, df)
        if name:
            person_today = today_df[today_df["Name"].str.lower() == name.lower()]
            if not person_today.empty:
                status = person_today.iloc[0].get("Status", "Present")
                return f"Yes, {name}'s attendance is marked today. Status: {status}"
            else:
                return f"No, {name}'s attendance has NOT been recorded today."

    # ── "Show attendance for [Name]" ──────────────────────
    if any(phrase in query for phrase in ["show attendance", "attendance of", "attendance for", "history"]):
        name = extract_name(query, df)
        if name:
            person_df = df[df["Name"].str.lower() == name.lower()]
            if person_df.empty:
                return f"No attendance records found for {name}."
            total = len(person_df)
            dates = person_df["Date"].tolist()
            return (f"{name} has {total} attendance record(s).\n"
                    f"Dates: {', '.join(str(d) for d in dates)}")

    # ── "Total records" ────────────────────────────────────
    if any(phrase in query for phrase in ["total records", "all records", "total attendance"]):
        return f"Total attendance records in the system: {len(df)}"

    # ── Fallback ───────────────────────────────────────────
    return (
        "Sorry, I didn't understand that. You can ask:\n"
        "• 'Who is present today?'\n"
        "• 'How many are present today?'\n"
        "• 'Is [Name] present?'\n"
        "• 'Show attendance for [Name]'\n"
        "• 'Total records'"
    )