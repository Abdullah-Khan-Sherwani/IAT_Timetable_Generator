# -*- coding: utf-8 -*-
import re
from itertools import product
from datetime import datetime, time
from typing import Dict, List, Tuple
from rapidfuzz import process

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

def fuzzy_deduplicate(names, cutoff=85):
    unique = []
    mapping = {}
    for n in names:
        match, score, _ = process.extractOne(n, unique, score_cutoff=cutoff)
        if match:
            mapping[n] = match
        else:
            unique.append(n)
            mapping[n] = n
    return mapping

def normalize_course_name(name: str) -> str:
    if not isinstance(name, str):
        return name

    # Lowercase and trim
    name = name.strip().lower()
    
    # Normalize dashes and spaces
    name = re.sub(r"\s*-\s*", " ", name)
    name = re.sub(r"\s+", " ", name)
    
    # Remove repeated words (e.g., "linear algebra linear algebra")
    words = name.split()
    unique_words = []
    for w in words:
        if not unique_words or unique_words[-1] != w:
            unique_words.append(w)
    name = " ".join(unique_words)
    
    # Convert roman numerals I/II/III to numbers
    name = re.sub(r"\bi\b", "1", name)
    name = re.sub(r"\bii\b", "2", name)
    name = re.sub(r"\biii\b", "3", name)

    # Title case for final presentation
    return name.title()


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
    """
    Google Sheet layout:
      time | [Mon/Wed 5 cols] | [Tue/Thu 5 cols] | [Fri/Sat 5 cols]
    The sheet often omits either the time (carried visually) or repeats course rows.
    We therefore:
      - carry the last seen time forward,
      - carry the last non-empty block forward for each day-pair block.
    """
    ROW_START = 7          # 0-based (row 8)
    ROW_END   = 384        # 0-based (row 385)
    TIME_COL  = 0

    BLOCKS = [
        dict(cols=[1,  2,  3,  4,  5],  days=("Mon", "Wed")),
        dict(cols=[6,  7,  8,  9, 10],  days=("Tue", "Thu")),
        dict(cols=[11, 12, 13, 14, 15], days=("Fri", "Sat")),
    ]

    tidy_rows = []

    def emit_rows(block_vals, daypair, start_t, end_t):
        if not block_vals:
            return
        # Are all 5 cols blank?
        if all(pd.isna(x) or (isinstance(x, str) and x.strip() == "") for x in block_vals):
            return

        course_name_raw = str(block_vals[0]).strip() if block_vals[0] else ""
        class_prog      = str(block_vals[1]).strip() if block_vals[1] else ""
        room            = str(block_vals[2]).strip() if block_vals[2] else ""
        ums             = str(block_vals[3]).strip() if block_vals[3] else ""
        teacher         = str(block_vals[4]).strip() if block_vals[4] else ""

        course_name_clean = normalize_course_name(course_name_raw)
        course_code = course_name_clean

        sec_guess = re.findall(r"(sec[^,\s]*)", course_name_raw, flags=re.IGNORECASE)
        if not sec_guess:
            sec_guess = re.findall(r"(sec[^,\s]*)", class_prog, flags=re.IGNORECASE)
        section = sec_guess[0] if sec_guess else None

        time_key = ""
        if isinstance(start_t, time) and isinstance(end_t, time):
            time_key = f"{start_t.strftime('%H:%M')}-{end_t.strftime('%H:%M')}"
        teacher_key = teacher if teacher else "Unknown"

        for d in daypair:
            # Build a single days_key for the block:
            days_key = "/".join(daypair)   # e.g. "Mon/Wed" or "Tue/Thu"
            time_key = f"{start_t.strftime('%H:%M')}-{end_t.strftime('%H:%M')}" if start_t and end_t else ""
            teacher_key = teacher or "Unknown"
            section_id = f"{course_code}|{teacher_key}|{days_key}|{time_key}"

            tidy_rows.append(
                dict(
                    course_code=course_code,
                    course_name=course_name_clean,
                    section=section,
                    section_id=section_id,
                    class_prog=class_prog,
                    ums=ums,
                    teacher=teacher,
                    day=d,
                    start=start_t,
                    end=end_t,
                    room=room,
                )
            )

    current_start, current_end = None, None
    # keep last non-empty block values for each of the 3 blocks
    last_blocks = [None, None, None]

    for i in range(ROW_START, min(ROW_END + 1, len(df))):
        # Update time if present; otherwise keep previous (carry forward)
        time_str = df.iat[i, TIME_COL] if TIME_COL < df.shape[1] else None
        if isinstance(time_str, str) and time_str.strip():
            st_, et_ = parse_time_range(time_str)
            if st_ and et_:
                current_start, current_end = st_, et_

        # Only skip rows until we find the first valid time
        if not (current_start and current_end):
            # But still cache any non-empty blocks so they can be used once time appears
            for bi, block in enumerate(BLOCKS):
                if max(block["cols"]) >= df.shape[1]:
                    continue
                vals = [df.iat[i, c] for c in block["cols"]]
                if not all(pd.isna(x) or str(x).strip() == "" for x in vals):
                    last_blocks[bi] = vals
            continue

        # We have a valid time -> emit for each block, using last non-empty if this row's is blank
        for bi, block in enumerate(BLOCKS):
            if max(block["cols"]) >= df.shape[1]:
                continue
            vals = [df.iat[i, c] for c in block["cols"]]
            if all(pd.isna(x) or str(x).strip() == "" for x in vals):
                vals = last_blocks[bi]
            else:
                last_blocks[bi] = vals
            if vals is not None:
                emit_rows(vals, block["days"], current_start, current_end)

    out = pd.DataFrame(tidy_rows)

    # DEBUG (uncomment to verify you're now seeing Ahsan Jawed)
    # st.write("Teachers found:", sorted(out['teacher'].dropna().unique()))
    # st.write(out[out['teacher'].str.contains("ahsan", case=False, na=False)])

    return out

# ------------------------------
# Clash detection
def overlaps(a, b) -> bool:
    if a["day"] != b["day"]:
        return False
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])


def schedule_is_valid(meetings: List[dict]) -> bool:
    for i in range(len(meetings)):
        for j in range(i + 1, len(meetings)):
            if meetings[i].get("_key") == meetings[j].get("_key"):
                continue
            if overlaps(meetings[i], meetings[j]):
                return False
    return True

# ------------------------------
# Build course_groups: course_code -> list of "sections", where each section is a list of meetings
# (because a section might meet multiple times a week).
def build_course_groups(df: pd.DataFrame) -> Dict[str, List[List[dict]]]:
    groups: Dict[str, List[List[dict]]] = {}

    if "section_id" not in df.columns:
        # fallback if user loaded a tidy CSV without our section_id — approximate
        df = df.assign(
            section_id=(df["course_code"].astype(str) + "|" +
                        df["teacher"].fillna("NA").astype(str) + "|" +
                        df["start"].astype(str) + "-" + df["end"].astype(str))
        )

    for (course_code, section_id), g in df.groupby(["course_code", "section_id"], dropna=False):
        key = f"{course_code}::{section_id}"
        meetings = []
        for _, row in g.iterrows():
            rec = row.to_dict()
            rec["_key"] = key  # tag to avoid intra-section clashes
            meetings.append(rec)
        groups.setdefault(str(course_code), []).append(meetings)

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
2. Paste the CSV export URL(s):  
   https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>  
   (one per worksheet/tab) below.
3. Pick layout type:
   - IBA-like layout (Mon/Wed | Tue/Thu | Fri/Sat blocks per row)
   - Already normalized (columns: course_code, section/section_id, day, start, end, ...)
4. Select courses (and optionally teachers) and generate all clash‑free schedules.
        """
    )

    csv_urls_input = st.text_area(
        "CSV export URL(s) - one per line",
        value="",
        placeholder="Paste one or more CSV export links here...",
        key="csv_urls_input",
    )
    if not csv_urls_input.strip():
        st.info("Paste at least one CSV URL to continue.")
        st.stop()

    csv_urls = [u.strip() for u in csv_urls_input.splitlines() if u.strip()]

    layout_type = st.radio(
        "Sheet layout",
        ["IBA-like", "Already normalized"],
        index=0,
        key="layout_type",
    )

    # ---------- Load & normalize ----------
    with st.spinner("Loading & normalizing ..."):
        tidy_all = []
        for url in csv_urls:
            raw = pd.read_csv(url, header=None if layout_type.startswith("IBA") else 0)
            if layout_type.startswith("IBA"):
                tidy = normalize_from_iba_like_layout(raw)
            else:
                tidy = normalize_already_tidy(raw)
            tidy_all.append(tidy)  # <--- important!

        df = pd.concat(tidy_all, ignore_index=True)
        df = df.dropna(subset=["start", "end", "day"])
        df["start"] = df["start"].apply(lambda t: t if isinstance(t, time) else t)
        df["end"]   = df["end"].apply(lambda t: t if isinstance(t, time) else t)

    st.subheader("Normalized data (preview)")
    st.dataframe(df.head(1000))


    # Debug
    st.write("Row 8 raw:", raw.iloc[6].tolist())
    # st.write("Row 8 raw:", raw.iloc[6].tolist())
    # st.write("Row 55 raw:", raw.iloc[53].tolist())
    # st.write("Row 102 raw:", raw.iloc[100].tolist())
    # st.write("Row 150 raw:", raw.iloc[148].tolist())
    # st.write("Row 197 raw:", raw.iloc[195].tolist())
    # st.write("Row 244 raw:", raw.iloc[242].tolist())
    # st.write("Row 291 raw:", raw.iloc[289].tolist())

# ---------- Course selection ----------
    courses = sorted([
    c for c in df["course_code"].astype(str).unique()
    if "lab" not in c.lower()
    ])

    chosen = st.multiselect("Pick your courses", courses, key="pick_courses")

    if not chosen:
        st.stop()

    # Define time slots
    time_slots = [
        ("8:30 AM", "9:45 AM"),
        ("10:00 AM", "11:15 AM"),
        ("11:30 AM", "12:45 PM"),
        ("1:00 PM", "2:15 PM"),
        ("2:30 PM", "3:45 PM"),
        ("4:00 PM", "5:15 PM"),
        ("5:30 PM", "6:45 PM")
    ]

    # Convert time strings to time objects for comparison
    time_slot_objects = []
    for start_str, end_str in time_slots:
        start_time = datetime.strptime(start_str, "%I:%M %p").time()
        end_time = datetime.strptime(end_str, "%I:%M %p").time()
        time_slot_objects.append((start_time, end_time))

    # Create time slot labels for display
    time_slot_labels = [f"{start} to {end}" for start, end in time_slots]

    # ---------- Optional day filters ----------
    day_options = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    day_filters: Dict[str, set] = {}
    with st.expander("Optional: filter days per course"):
        for c in chosen:
            selected_days = st.multiselect(
                f"Allowed days for {c} (leave empty = all)",
                day_options,
                default=day_options,
                key=f"day_filter_{c}",
            )
            day_filters[c] = set(selected_days) if selected_days else set(day_options)

    # ---------- Optional time slot filters ----------
    time_filters: Dict[str, set] = {}
    with st.expander("Optional: filter time slots per course"):
        for c in chosen:
            selected_slots = st.multiselect(
                f"Allowed time slots for {c} (leave empty = all)",
                time_slot_labels,
                default=time_slot_labels,
                key=f"time_filter_{c}",
            )
            # Store the corresponding time objects for filtering
            selected_indices = [i for i, label in enumerate(time_slot_labels) if label in selected_slots]
            time_filters[c] = {time_slot_objects[i] for i in selected_indices} if selected_slots else set(time_slot_objects)

    # Apply filters
    mask = pd.Series(True, index=df.index)
    for c in chosen:
        allowed_days = day_filters.get(c, set(day_options))
        allowed_times = time_filters.get(c, set(time_slot_objects))
        
        # Get all section_ids for this course that have:
        # 1. ANY day not in allowed_days, OR
        # 2. ANY time slot not in allowed_times
        bad_sections = set()
        for section_id, group in df[df.course_code == c].groupby('section_id'):
            section_days = set(group['day'].unique())
            time_violation = False
            
            # Check if any meeting time falls outside allowed time slots
            for _, row in group.iterrows():
                start = row['start']
                end = row['end']
                
                # Check if this meeting time matches any allowed time slot
                time_match = False
                for (slot_start, slot_end) in allowed_times:
                    if start == slot_start and end == slot_end:
                        time_match = True
                        break
                
                if not time_match:
                    time_violation = True
                    break
            
            if not section_days.issubset(allowed_days) or time_violation:
                bad_sections.add(section_id)
        
        # Keep rows that either:
        # 1. Aren't for this course, OR
        # 2. Are for this course but not in bad_sections
        mask &= (
            (df.course_code != c) |
            ((df.course_code == c) & (~df.section_id.isin(bad_sections)))
        )
    
    filtered_df = df[mask]

    # ---------- Optional teacher filters ----------
    teacher_filters: Dict[str, set] = {}
    with st.expander("Optional: filter teachers per course"):
        for c in chosen:
            teachers = sorted(
                filtered_df.loc[filtered_df.course_code.astype(str) == c, "teacher"]
                  .dropna()
                  .astype(str)
                  .unique()
            )
            sel = st.multiselect(
                f"Allowed teachers for {c} (leave empty = all)",
                teachers,
                default=teachers,
                key=f"teacher_filter_{c}",
            )
            teacher_filters[c] = set(sel) if sel else set(teachers)
            

    # Apply all filters to build the dataframe we'll actually solve on
    mask = pd.Series(False, index=filtered_df.index)
    for c in chosen:
        allowed = teacher_filters.get(c, set())
        # allow rows with teacher NaN OR in allowed list
        mask |= (
            (filtered_df.course_code.astype(str) == c) &
            (filtered_df.teacher.isna() | filtered_df.teacher.astype(str).isin(list(allowed))))
    use_df = filtered_df[mask]

    # ---------- Section manager & lab linker ----------
    # 1) SECTION MANAGER
    st.subheader("🔧 Manage Sections")
    # Build a mapping: course_code -> list of section_ids
    available = {}
    for course, group in use_df.groupby("course_code"):
        available[course] = sorted(
            {m["section_id"] for m in group.to_dict("records")}
        )

    # Let user pick which section_ids they actually want considered
    section_selection = {}
    for course in chosen:
        sec_list = available.get(course, [])
        # default = all
        default = sec_list.copy()
        picked = st.multiselect(
            f"{course}: pick sections",
            sec_list,
            default=default,
            key=f"section_mgr_{course}"
        )
        section_selection[course] = set(picked)

    # Filter use_df down to only those section_ids
    mask2 = pd.Series(False, index=use_df.index)
    for course, secs in section_selection.items():
        mask2 |= (
            (use_df.course_code == course) &
            (use_df.section_id.isin(secs))
        )
    use_df = use_df[mask2]

    # 2) LAB LINKER - REVISED (only for courses that actually have labs)
    st.subheader("🔗 Link Labs to Specific Class Sections")

    # detect all lab courses (e.g. "PHY LAB", "CS LAB", etc.)
    lab_courses = sorted({c for c in df.course_code.unique() if "LAB" in c.upper()})

    # find base course codes that have matching lab course codes
    # e.g., if "PHY101" has a "PHY101 LAB", consider PHY101 as having a lab
    base_courses_with_labs = set()
    for lab in lab_courses:
        for course in df.course_code.unique():
            if course in lab and course != lab:
                base_courses_with_labs.add(course)

    # persistent session state
    if "lab_links_by_section" not in st.session_state:
        st.session_state.lab_links_by_section = {}  # section_id -> list of lab course_codes

    with st.expander("Set up lab → class section links"):
        for course in chosen:
            if course not in base_courses_with_labs:
                continue  # skip if this course has no matching lab

            for section in section_selection.get(course, []):
                key = f"{course}::{section}"
                linked_labs = st.multiselect(
                    f"Link labs to section {section} of {course}:",
                    [lab for lab in lab_courses if course in lab],
                    default=st.session_state.lab_links_by_section.get(key, []),
                    key=f"lablink_section_{key}"
                )
                st.session_state.lab_links_by_section[key] = linked_labs

    # Collect lab rows linked to specific sections
    lab_rows = []
    for section_key, linked_labs in st.session_state.lab_links_by_section.items():
        course, section_id = section_key.split("::", 1)
        for lab_code in linked_labs:
            matched = df[(df.course_code == lab_code)].copy()
            if not matched.empty:
                matched["linked_section_id"] = section_id
                matched["linked_to_course"] = course
                lab_rows.append(matched)

    if lab_rows:
        lab_df = pd.concat(lab_rows, ignore_index=True)
        lab_df["section_id"] = lab_df["course_code"] + "::" + lab_df["linked_section_id"]
    else:
        lab_df = pd.DataFrame(columns=df.columns)

    use_df = pd.concat([use_df, lab_df], ignore_index=True)



    # Now build groups & generate as before:
    groups = build_course_groups(use_df)
    schedules = generate_all_schedules(groups)

    # ---------- Generate ----------
    max_show = st.number_input("Max schedules to show", 1, 2000, value=50, key="max_show")

    if st.button("Generate schedules", key="gen"):
        groups = build_course_groups(use_df)        # groups by (course_code, section_id)
        schedules = generate_all_schedules(groups)  # skips intra-section clashes

        st.success(f"Found {len(schedules)} clash-free schedules.")
        for i, sched in enumerate(schedules[:max_show]):
            st.markdown(f"### Schedule #{i+1}")
            show = (
                pd.DataFrame(sched)[
                    ["course_code", "section", "section_id", "course_name",
                     "teacher", "day", "start", "end", "room"]
                ]
                .sort_values(by=["day", "start"])
            )
            st.dataframe(show)

        if len(schedules) > max_show:
            st.info(f"Showing first {max_show} only.")



if __name__ == "__main__":
    main()
