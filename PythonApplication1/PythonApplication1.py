# -*- coding: utf-8 -*-
import re
from itertools import product
from datetime import datetime, time
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


# ==============================
# --------- CONFIG -------------
# ==============================
# If your sheet is PUBLIC, you can read each worksheet/tab with:
# https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>
#
# Paste those CSV URLs in the textbox (one per line) and choose "IBA-like layout"
# OR provide a single "already-normalized" CSV and choose "Normalized layout".


# ------------------------------
# Utility: parse "8:30 AM to 9:45 AM" -> (08:30, 09:45)
def parse_time_range(s: str) -> Tuple[time, time]:
    if pd.isna(s) or not isinstance(s, str):
        return None, None
    m = re.search(r'(\d{1,2}:\d{2}\s*[AP]M)\s*to\s*(\d{1,2}:\d{2}\s*[AP]M)', s, re.IGNORECASE)
    if not m:
        return None, None
    start = datetime.strptime(m.group(1).upper().replace(" ", ""), "%I:%M%p").time()
    end = datetime.strptime(m.group(2).upper().replace(" ", ""), "%I:%M%p").time()
    return start, end


# ------------------------------
# Heuristic parser for your Google Sheet:
# pattern repeated 3 times per row:
# time | (Mon/Wed: Course, Class&Program, Room, UMS, Teacher)
#      | (Tue/Thu: Course, Class&Program, Room, UMS, Teacher)
#      | (Fri/Sat: Course, Class&Program, Room, UMS, Teacher)
def normalize_from_iba_like_layout(df: pd.DataFrame) -> pd.DataFrame:
    ROW_START = 7          # 0-based (row 8 in the sheet)
    ROW_END   = 384        # 0-based (row 385 in the sheet)
    TIME_COL  = 0

    BLOCKS = [
        dict(cols=[1, 2, 3, 4, 5],      days=("Mon", "Wed")),
        dict(cols=[6, 7, 8, 9, 10],     days=("Tue", "Thu")),
        dict(cols=[11, 12, 13, 14, 15], days=("Fri", "Sat")),
    ]

    tidy_rows = []

    def emit_rows(block_vals, daypair, start_t, end_t):
        if all(pd.isna(x) or (isinstance(x, str) and x.strip() == "") for x in block_vals):
            return

        course_name = str(block_vals[0]).strip() if not pd.isna(block_vals[0]) else ""
        class_prog  = str(block_vals[1]).strip() if not pd.isna(block_vals[1]) else ""
        room        = str(block_vals[2]).strip() if not pd.isna(block_vals[2]) else ""
        teacher     = str(block_vals[4]).strip() if not pd.isna(block_vals[4]) else ""

        # crude guesses; adjust if you have explicit columns for these
        course_code_guess = re.findall(r'\b[A-Z]{2,}\s*\d*\b', course_name)
        course_code = course_code_guess[0] if course_code_guess else course_name

        section_guess = re.findall(r'(Sec[^,\s]*)', course_name, flags=re.IGNORECASE)
        if not section_guess:
            section_guess = re.findall(r'(Sec[^,\s]*)', class_prog, flags=re.IGNORECASE)
        section = section_guess[0] if section_guess else None

        for d in daypair:
            tidy_rows.append(
                dict(
                    course_code=course_code,
                    section=section,
                    course_name=course_name,
                    day=d,
                    start=start_t,
                    end=end_t,
                    room=room,
                    teacher=teacher
                )
            )

    current_start, current_end = None, None

    for i in range(ROW_START, min(ROW_END + 1, len(df))):
        # timing column
        time_str = df.iat[i, TIME_COL] if TIME_COL < df.shape[1] else None
        if isinstance(time_str, str):
            st_, et_ = parse_time_range(time_str)
            if st_ and et_:
                current_start, current_end = st_, et_

        if not current_start or not current_end:
            continue

        for block in BLOCKS:
            if max(block["cols"]) >= df.shape[1]:
                continue
            vals = [df.iat[i, c] for c in block["cols"]]
            emit_rows(vals, block["days"], current_start, current_end)

    return pd.DataFrame(tidy_rows)


# ------------------------------
# If you already have a tidy CSV with columns:
# [course_code, section, course_name, day, start, end, room, teacher]
# you can just read it directly with this.
def normalize_already_tidy(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = [c.lower() for c in df.columns]

    # Try to be flexible with column names
    def get_col(name_candidates):
        for n in name_candidates:
            if n in cols_lower:
                return df.columns[cols_lower.index(n)]
        return None

    col_course = get_col(["course_code", "course", "code"])
    col_section = get_col(["section", "sec"])
    col_day = get_col(["day"])
    col_start = get_col(["start", "start_time", "from"])
    col_end = get_col(["end", "end_time", "to"])
    col_name = get_col(["course_name", "name"])
    col_room = get_col(["room", "room_no"])
    col_teacher = get_col(["teacher", "instructor", "professor"])

    need = [col_course, col_day, col_start, col_end]
    if any(x is None for x in need):
        raise ValueError("Your tidy CSV must at least contain course_code, day, start, end columns (case-insensitive).")

    def to_time(x):
        if isinstance(x, time):
            return x
        if pd.isna(x):
            return None
        s = str(x).strip()
        for fmt in ["%H:%M", "%I:%M%p", "%I:%M %p"]:
            try:
                return datetime.strptime(s, fmt).time()
            except:
                pass
        return None

    result = []
    for _, row in df.iterrows():
        result.append(dict(
            course_code=row[col_course],
            section=row[col_section] if col_section else None,
            course_name=row[col_name] if col_name else row[col_course],
            day=row[col_day],
            start=to_time(row[col_start]),
            end=to_time(row[col_end]),
            room=row[col_room] if col_room else None,
            teacher=row[col_teacher] if col_teacher else None,
        ))
    return pd.DataFrame(result)


# ------------------------------
# Clash detection
def overlaps(a, b) -> bool:
    if a["day"] != b["day"]:
        return False
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])


def schedule_is_valid(meetings: List[dict]) -> bool:
    for i in range(len(meetings)):
        for j in range(i + 1, len(meetings)):
            if overlaps(meetings[i], meetings[j]):
                return False
    return True


# ------------------------------
# Build course_groups: course_code -> list of "sections", where each section is a list of meetings
# (because a section might meet multiple times a week).
def build_course_groups(df: pd.DataFrame) -> Dict[str, List[List[dict]]]:
    groups = {}
    for (course_code, section), g in df.groupby(["course_code", "section"], dropna=False):
        meetings = g.to_dict("records")
        groups.setdefault(course_code, []).append(meetings)
    return groups


def generate_all_schedules(course_groups: Dict[str, List[List[dict]]]) -> List[List[dict]]:
    # Cartesian product: choose exactly 1 section for each course_code
    # Then flatten and test for clashes.
    if not course_groups:
        return []
    all_choices = product(*course_groups.values())
    valid = []
    for choice in all_choices:
        flat = [m for section in choice for m in section]
        if schedule_is_valid(flat):
            valid.append(flat)
    return valid


# ======================================================
# ---------------- Streamlit APP -----------------------
# ======================================================
def main():
    st.title("Timetable Generator (Streamlit, Python)")

    st.markdown(
        """
How to use:
1. Make your Google Sheet (tab) public or "anyone with the link can view".
2. Build the CSV export URL(s) like  
   https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>  
   (one per worksheet/tab) and paste them below (one per line).
3. Pick layout type:
   - IBA-like layout (your screenshot: Mon/Wed left, Tue/Thu middle, Fri/Sat right).
   - Already normalized (columns: course_code, section, day, start, end, ...).
4. Select courses and generate all clash-free schedules.
        """
    )

    layout_type = st.radio(
        "Sheet layout",
        ["IBA-like (screenshot style)", "Already normalized"],
        index=0
    )

    csv_urls_input = st.text_area(
        "CSV export URL(s) - one per line",
        value="",
        placeholder="Paste one or more CSV export links here..."
    )
    if not csv_urls_input.strip():
        st.info("Paste at least one CSV URL to continue.")
        st.stop()

    csv_urls = [u.strip() for u in csv_urls_input.splitlines() if u.strip()]

    with st.spinner("Loading & normalizing ..."):
        tidy_all = []
        for url in csv_urls:
            raw = pd.read_csv(url, header=None if layout_type == "IBA-like (screenshot style)" else 0)
            if layout_type.startswith("IBA"):
                tidy = normalize_from_iba_like_layout(raw)
            else:
                tidy = normalize_already_tidy(raw)
            tidy_all.append(tidy)

        df = pd.concat(tidy_all, ignore_index=True)
        df = df.dropna(subset=["start", "end", "day"])
        df["start"] = df["start"].apply(lambda t: t if isinstance(t, time) else t)
        df["end"] = df["end"].apply(lambda t: t if isinstance(t, time) else t)

    st.subheader("Normalized data (preview)")
    st.dataframe(df.head(100))

    courses = sorted(df["course_code"].astype(str).unique())
    chosen = st.multiselect("Pick your courses", courses)

    max_show = st.number_input("Max schedules to show", 1, 2000, value=50)
    if st.button("Generate schedules"):
        if not chosen:
            st.warning("Please pick at least one course.")
            st.stop()

        use_df = df[df.course_code.astype(str).isin(chosen)]
        groups = build_course_groups(use_df)
        schedules = generate_all_schedules(groups)

        st.success(f"Found {len(schedules)} clash-free schedules.")
        for i, sched in enumerate(schedules[:max_show]):
            st.markdown(f"### Schedule #{i+1}")
            show = pd.DataFrame(sched)[
                ["course_code", "section", "course_name", "day", "start", "end", "room", "teacher"]
            ].sort_values(by=["day", "start"])
            st.dataframe(show)

        if len(schedules) > max_show:
            st.info(f"Showing first {max_show} only.")


if __name__ == "__main__":
    main()
