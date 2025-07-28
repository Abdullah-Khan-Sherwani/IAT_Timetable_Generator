# -*- coding: utf-8 -*-
import re
from itertools import product
from datetime import datetime, time
from typing import Dict, List, Tuple
from rapidfuzz import process

import plotly.express as px
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def compute_schedules(use_df):
    """
    Build groups and generate clash-free schedules.
    Cached so it only reruns when `use_df` changes.
    """
    groups = build_course_groups(use_df)
    return generate_all_schedules(groups)

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

def render_weekly_grid(sched):
    # 1) Define the fixed IBA slots
    time_labels = [
        "08:30-09:45", "10:00-11:15", "11:30-12:45",
        "13:00-14:15", "14:30-15:45", "16:00-17:15",
        "17:30-18:45"
    ]
    days = ["Mon","Tue","Wed","Thu","Fri","Sat"]

    # 2) Build an empty grid
    grid = {slot: {d: "" for d in days} for slot in time_labels}

    # 3) Fill it from the schedule
    df_sched = pd.DataFrame(sched)
    for _, row in df_sched.iterrows():
        slot = f"{row['start'].strftime('%H:%M')}-{row['end'].strftime('%H:%M')}"
        d   = row["day"]
        if slot in grid and d in days:
            # you can tweak to show teacher, section_id, etc.
            grid[slot][d] = f"{row['course_code']}<br><small>{row['room']}</small>"

    # 4) Generate HTML table
    header = "<th style='border:1px solid #ddd;padding:4px'>Time</th>" + \
             "".join(f"<th style='border:1px solid #ddd;padding:4px'>{d}</th>" for d in days)
    rows = ""
    for slot in time_labels:
        row_html = f"<td style='border:1px solid #ddd;padding:4px'>{slot}</td>" + \
                   "".join(f"<td style='border:1px solid #ddd;padding:4px'>{grid[slot][d]}</td>"
                           for d in days)
        rows += f"<tr>{row_html}</tr>"

    table = f"""
    <table style='border-collapse: collapse; width: 100%; text-align: center;'>
      <thead><tr>{header}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """

    st.subheader("📋 Weekly Timetable Grid")
    st.write(table, unsafe_allow_html=True)

def show_calendar(sched):
    # Build DataFrame
    df_sched = pd.DataFrame(sched)

    # Anchor times to a dummy date so we can plot just times of day
    def to_dt(t):
        return datetime(2025,1,1,t.hour,t.minute) if hasattr(t, "hour") else None

    df_sched["start_dt"] = df_sched["start"].apply(to_dt)
    df_sched["end_dt"]   = df_sched["end"].apply(to_dt)

    # Make the timeline
    fig = px.timeline(
        df_sched,
        x_start="start_dt",
        x_end="end_dt",
        y="day",
        color="course_code",
        text="course_code",
        hover_data=["room","teacher"]
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=["Mon","Tue","Wed","Thu","Fri","Sat"]
    )
    fig.update_xaxes(tickformat="%H:%M", dtick=3600000)
    fig.update_layout(height=500, margin={"l":0,"r":0,"t":30,"b":0}, showlegend=False)

    st.subheader("📅 Weekly Timetable")
    st.plotly_chart(fig, use_container_width=True)


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
    st.title("IAT Timetable Generator")

    st.markdown(
        """
How to use:
1. Make your Google Sheet (tab) public or “anyone with the link can view.”
2. Paste the CSV export URL(s):  
   `https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>`  
   (one per worksheet/tab) below.
3. Pick layout type:
   - IBA‑like (Mon/Wed | Tue/Thu | Fri/Sat blocks per row)
   - Already normalized (columns: course_code, section/section_id, day, start, end, …)
4. Select courses (and optionally teachers, days, times) and generate all clash‑free schedules.
        """
    )

    # — Input CSV URLs —
    csv_urls = [u.strip() for u in st.text_area(
        "CSV export URL(s) – one per line",
        placeholder="Paste one or more CSV export links here…",
        key="csv_urls_input"
    ).splitlines() if u.strip()]
    if not csv_urls:
        st.info("Paste at least one CSV URL to continue.")
        st.stop()

    # — Layout choice —
    layout_type = st.radio(
        "Sheet layout",
        ["IBA-like", "Already normalized"],
        index=0,
        key="layout_type"
    )

    # — Load & normalize —
    with st.spinner("Loading & normalizing …"):
        tidy_all = []
        for url in csv_urls:
            raw = pd.read_csv(url, header=None if layout_type.startswith("IBA") else 0)
            if layout_type.startswith("IBA"):
                tidy = normalize_from_iba_like_layout(raw)
            else:
                tidy = normalize_already_tidy(raw)
            tidy_all.append(tidy)
        df = pd.concat(tidy_all, ignore_index=True)
        df = df.dropna(subset=["start", "end", "day"])
        df["start"] = df["start"].apply(lambda t: t if isinstance(t, time) else t)
        df["end"]   = df["end"].apply(lambda t: t if isinstance(t, time) else t)

        st.subheader("Normalized data (preview)")
    st.dataframe(df.head(1000))

    # — Optional: Manually unify duplicate courses —
    # — Optional: Merge duplicate courses manually —
    if "course_merge_map" not in st.session_state:
        st.session_state.course_merge_map = {}

    with st.expander("🔄 Merge duplicate course entries"):
        all_codes = sorted(df["course_code"].astype(str).unique())
        remaining = [c for c in all_codes if c not in st.session_state.course_merge_map]

        with st.form("merge_form"):
            to_merge = st.multiselect("Select course variants to merge", remaining)
            default_canon = to_merge[0] if to_merge else ""
            canonical = st.text_input("Unified course code/name", value=default_canon)
            submitted = st.form_submit_button("Add merge")

            if submitted:
                if to_merge and canonical.strip():
                    for v in to_merge:
                        st.session_state.course_merge_map[v] = canonical.strip()
                    # apply right away:
                    df["course_code"] = df["course_code"].map(
                        lambda x: st.session_state.course_merge_map.get(x, x)
                    )
                    df["course_name"] = df["course_name"].map(
                        lambda x: st.session_state.course_merge_map.get(x, x)
                    )
                    st.success(f"Merged {to_merge} → **{canonical.strip()}**")
                else:
                    st.warning("Please select at least one variant *and* enter a canonical name.")

    # Now *outside* the form*, show the full list of merges:
        # — Now that course_merge_map is up‑to‑date, apply it on every render —
    if st.session_state.course_merge_map:
        df["course_code"] = df["course_code"].map(
            lambda x: st.session_state.course_merge_map.get(x, x)
        )
        df["course_name"] = df["course_name"].map(
            lambda x: st.session_state.course_merge_map.get(x, x)
        )

    # — Course selection (exclude labs) —
    courses = sorted([
        c for c in df["course_code"].astype(str).unique()
        if "lab" not in c.lower()
    ])
    chosen = st.multiselect("Pick your courses", courses, key="pick_courses")
    if not chosen:
        st.stop()

    # — Define IBA time‐slots —
    time_slots = [
        ("8:30 AM", "9:45 AM"), ("10:00 AM", "11:15 AM"),
        ("11:30 AM", "12:45 PM"), ("1:00 PM", "2:15 PM"),
        ("2:30 PM", "3:45 PM"), ("4:00 PM", "5:15 PM"),
        ("5:30 PM", "6:45 PM")
    ]
    # convert to time objects
    slot_objs = []
    for s, e in time_slots:
        slot_objs.append((
            datetime.strptime(s, "%I:%M %p").time(),
            datetime.strptime(e, "%I:%M %p").time()
        ))
    slot_labels = [f"{s} to {e}" for s, e in time_slots]

    # — Optional: Day filters per course —
    day_options = ["Mon","Tue","Wed","Thu","Fri","Sat"]
    day_filters = {}
    with st.expander("Optional: filter days per course"):
        for c in chosen:
            sel = st.multiselect(
                f"Allowed days for {c} (leave empty = all)",
                day_options,
                default=day_options,
                key=f"day_filter_{c}"
            )
            day_filters[c] = set(sel) if sel else set(day_options)

    # — Optional: Time‐slot filters per course —
    time_filters = {}
    with st.expander("Optional: filter time slots per course"):
        for c in chosen:
            sel = st.multiselect(
                f"Allowed time slots for {c} (leave empty = all)",
                slot_labels,
                default=slot_labels,
                key=f"time_filter_{c}"
            )
            if sel:
                idxs = [i for i,label in enumerate(slot_labels) if label in sel]
                time_filters[c] = {slot_objs[i] for i in idxs}
            else:
                time_filters[c] = set(slot_objs)

    # — Apply day & time filters —
    mask = pd.Series(True, index=df.index)
    for c in chosen:
        bad_secs = set()
        sub = df[df.course_code == c]
        for sid, grp in sub.groupby("section_id"):
            days_ok = set(grp["day"]).issubset(day_filters[c])
            times_ok = all(
                (row["start"], row["end"]) in time_filters[c]
                for _, row in grp.iterrows()
            )
            if not (days_ok and times_ok):
                bad_secs.add(sid)
        mask &= ~((df.course_code == c) & df.section_id.isin(bad_secs))
    filtered_df = df[mask]

    # — Optional: Teacher filters —
    teacher_filters = {}
    with st.expander("Optional: filter teachers per course"):
        for c in chosen:
            tlist = sorted(
                filtered_df[filtered_df.course_code == c]["teacher"]
                .dropna().astype(str).unique()
            )
            sel = st.multiselect(
                f"Allowed teachers for {c} (leave empty = all)",
                tlist,
                default=tlist,
                key=f"teacher_filter_{c}"
            )
            teacher_filters[c] = set(sel) if sel else set(tlist)

    # — Apply teacher filters —
    mask = pd.Series(False, index=filtered_df.index)
    for c in chosen:
        mask |= (
            (filtered_df.course_code == c) &
            (filtered_df.teacher.isna() |
             filtered_df.teacher.isin(teacher_filters[c]))
        )
    use_df = filtered_df[mask]

    # — Section manager —
    st.subheader("🔧 Manage Sections")
    available = {
        course: sorted({r["section_id"] for r in grp.to_dict("records")})
        for course, grp in use_df.groupby("course_code")
    }
    section_selection = {}
    for course in chosen:
        secs = available.get(course, [])
        picked = st.multiselect(
            f"{course}: pick sections",
            secs,
            default=secs,
            key=f"section_mgr_{course}"
        )
        section_selection[course] = set(picked)
    mask2 = pd.Series(False, index=use_df.index)
    for course, secs in section_selection.items():
        mask2 |= (
            (use_df.course_code == course) &
            (use_df.section_id.isin(secs))
        )
    use_df = use_df[mask2]

    # — Lab linker per lecture‑section (by UMS) —
    st.subheader("🔗 Attach labs to each lecture section (via UMS)")

    lab_df = df[df["course_name"].str.contains("lab", case=False, na=False)]
    labs_by_ums = (
        lab_df.groupby("ums")["section_id"]
        .unique().apply(list)
        .to_dict()
    )

    if "lab_links" not in st.session_state:
        st.session_state.lab_links = {}

    with st.expander("Pick which labs go with each lecture section"):
        for course, secs in section_selection.items():
            for lecture_sec in secs:
                # pull only labs sharing that UMS
                lecture_ums = use_df.loc[use_df.section_id == lecture_sec, "ums"]
                if lecture_ums.empty:
                    continue
                options = labs_by_ums.get(lecture_ums.iat[0], [])
                default = st.session_state.lab_links.get(lecture_sec, options)
                picked = st.multiselect(
                    f"Labs for {course} [{lecture_sec}]:",
                    options=options,
                    default=default,
                    key=f"lablink_{lecture_sec}",
                    help="Only labs with the same UMS will appear here."
                )
                st.session_state.lab_links[lecture_sec] = picked

    # merge the picked labs into use_df under the same lecture section
    extra = []
    for lecture_sec, lab_sids in st.session_state.lab_links.items():
        if not lab_sids:
            continue

        lecture_rows = use_df[use_df.section_id == lecture_sec]
        if lecture_rows.empty:
            continue  # nothing to attach to!

        parent_course = lecture_rows["course_code"].iat[0]
        labs = df[df["section_id"].isin(lab_sids)].copy()
        labs["course_code"] = parent_course
        labs["section_id"]  = lecture_sec
        extra.append(labs)

    if extra:
        use_df = pd.concat([use_df, pd.concat(extra, ignore_index=True)], ignore_index=True)

        # 4) generate
        max_show = st.number_input("Max schedules to show", 1, 2000, value=50, key="max_show")
        with st.spinner("Computing all clash‑free schedules…"):
            schedules = compute_schedules(sched_df)
        st.success(f"Found {len(schedules)} clash‑free schedules.")

        for idx, sched in enumerate(schedules[:max_show], 1):
            st.markdown(f"### Schedule #{idx}")
            render_weekly_grid(sched)

        if len(schedules) > max_show:
            st.info(f"Showing first {max_show} only.")


    # # ---------- Generate ----------
    # max_show = st.number_input("Max schedules to show", 1, 2000, value=50, key="max_show")

    # if st.button("Generate schedules", key="gen"):
    #     st.success(f"Found {len(schedules)} clash-free schedules.")
    #     for i, sched in enumerate(schedules[:max_show]):
    #         st.markdown(f"### Schedule #{i+1}")
    #         show = (
    #             pd.DataFrame(sched)[
    #                 ["course_code", "section", "section_id", "course_name",
    #                  "teacher", "day", "start", "end", "room"]
    #             ]
    #             .sort_values(by=["day", "start"])
    #         )
            # st.dataframe(show) # Simple table view
            # show_calendar(sched) # Plotly calendar view
    # # better view
    # if schedules:
    #     st.subheader("📋 All Weekly Timetable Grids")
    #     for i, sched in enumerate(schedules):
    #         st.markdown(f"#### Schedule #{i+1}")
    #         render_weekly_grid(sched)
    # else:
    #     st.info("No schedules to display yet.")

    #     if len(schedules) > max_show:
    #         st.info(f"Showing first {max_show} only.")

if __name__ == "__main__":
    main()