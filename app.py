"""
Syslog Interval Viewer (Streamlit)

Summary:
- Queries `journalctl` for a selected date/time range.
- Chunks logs into fixed-size intervals.
- Displays first log + count per interval with a View checkbox.
- Opens a modal with full logs for the selected interval.

Usage:
1) Pick date range and start/end times in the sidebar.
2) Set interval seconds (default 10).
3) Click Fetch logs.
4) Click the 🔍 checkbox for an interval to open details.

Notes:
- Requires Linux with systemd/journalctl access.
- Logs are cached in-session to minimize repeated journalctl calls.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import streamlit as st
import pandas as pd
from html import escape

# Parsed journal entry with timestamp and derived message.
@dataclass(frozen=True)
class LogEntry:
    ts: datetime
    message: str
    cursor: Optional[str]
    raw: Dict

# Fixed interval row with optional first log entry.
@dataclass(frozen=True)
class IntervalRow:
    index: int
    start: datetime
    end: datetime
    first_log: Optional[LogEntry]


# Resolve the system's local timezone for display.
def local_tz() -> timezone:
    return datetime.now().astimezone().tzinfo  # type: ignore[return-value]


# Format a full timestamp for UI display (YYYY-MM-DD + 12-hour time).
def format_dt(dt: datetime) -> str:
    return dt.astimezone(local_tz()).strftime("%Y-%m-%d %I:%M:%S %p")


# Date-only formatting for table labels.
def format_date(dt: datetime) -> str:
    return dt.astimezone(local_tz()).strftime("%Y-%m-%d")


# Time-only formatting for table labels.
def format_time(dt: datetime) -> str:
    return dt.astimezone(local_tz()).strftime("%I:%M:%S %p")


# Build the interval label, suppressing repeated dates for cleaner display.
def format_interval_label(interval: IntervalRow, last_date: Optional[str]) -> Tuple[str, Optional[str]]:
    start_date = format_date(interval.start)
    end_date = format_date(interval.end)
    start_time = format_time(interval.start)
    end_time = format_time(interval.end)

    if start_date == end_date:
        if last_date == start_date:
            label = f"{start_time} → {end_time}"
        else:
            label = f"{start_date} {start_time} → {end_time}"
        return label, start_date

    label = f"{start_date} {start_time} → {end_date} {end_time}"
    return label, end_date


# Convert a seconds offset from midnight into a time object.
def seconds_to_time(total_seconds: int) -> time:
    total_seconds = max(0, min(86399, total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return time(hours, minutes, seconds)


# Clean log text to avoid control chars or non-printable bytes breaking UI rendering.
def sanitize_text(text: str, max_len: Optional[int] = None) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    cleaned = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if max_len and len(cleaned) > max_len:
        return cleaned[: max_len - 1] + "…"
    return cleaned



# Build a readable log message from journal fields.
def build_message(entry: Dict) -> str:
    parts = []
    if entry.get("SYSLOG_IDENTIFIER"):
        parts.append(entry["SYSLOG_IDENTIFIER"])
    if entry.get("_SYSTEMD_UNIT"):
        parts.append(entry["_SYSTEMD_UNIT"])
    if entry.get("_HOSTNAME"):
        parts.append(entry["_HOSTNAME"])
    header = " | ".join(parts)
    msg = entry.get("MESSAGE", "")
    return f"{header}: {msg}" if header else msg


# Parse JSON journal entry into LogEntry, returning None if invalid.
def parse_entry(entry: Dict) -> Optional[LogEntry]:
    ts_raw = entry.get("__REALTIME_TIMESTAMP")
    if not ts_raw:
        return None
    try:
        ts = datetime.fromtimestamp(int(ts_raw) / 1_000_000, tz=timezone.utc).astimezone(local_tz())
    except Exception:
        return None
    cursor = entry.get("__CURSOR")
    return LogEntry(ts=ts, message=build_message(entry), cursor=cursor, raw=entry)


# Execute journalctl for a time range and parse JSON output.
def run_journalctl(start: datetime, end: datetime) -> List[LogEntry]:
    start_str = start.astimezone(local_tz()).strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.astimezone(local_tz()).strftime("%Y-%m-%d %H:%M:%S")
    cmd = ["journalctl", "--output=json", "--since", start_str, "--until", end_str]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "journalctl failed")
    entries: List[LogEntry] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed = parse_entry(obj)
        if parsed:
            entries.append(parsed)
    entries.sort(key=lambda e: e.ts)
    return entries


# Merge overlapping/adjacent ranges for cache tracking.
def merge_ranges(ranges: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda r: r[0])
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


# Compute missing ranges given requested and already-cached ranges.
def subtract_ranges(requested: Tuple[datetime, datetime],
                    available: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    req_start, req_end = requested
    if req_start >= req_end:
        return []
    if not available:
        return [(req_start, req_end)]
    available = merge_ranges(available)
    missing: List[Tuple[datetime, datetime]] = []
    cursor = req_start
    for start, end in available:
        if end <= cursor:
            continue
        if start > cursor:
            missing.append((cursor, min(start, req_end)))
            cursor = start
        if cursor >= req_end:
            break
        if end > cursor:
            cursor = end
    if cursor < req_end:
        missing.append((cursor, req_end))
    return [(s, e) for s, e in missing if s < e]


# Merge and dedupe log entries using journal cursor when available.
def dedupe_entries(entries: Iterable[LogEntry],
                   existing: List[LogEntry],
                   existing_cursors: set) -> List[LogEntry]:
    merged = list(existing)
    seen = set(existing_cursors)
    for entry in entries:
        key = entry.cursor or f"{entry.ts.timestamp()}|{entry.message}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
    merged.sort(key=lambda e: e.ts)
    existing_cursors.clear()
    existing_cursors.update(seen)
    return merged


# Filter cached entries for the requested range.
def filter_entries(entries: List[LogEntry], start: datetime, end: datetime) -> List[LogEntry]:
    return [e for e in entries if start <= e.ts <= end]


# Build fixed-size interval list between start and end.
def build_intervals(start: datetime, end: datetime, seconds: int) -> List[IntervalRow]:
    if seconds <= 0:
        return []
    intervals: List[IntervalRow] = []
    idx = 0
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + timedelta(seconds=seconds), end)
        intervals.append(IntervalRow(index=idx, start=cursor, end=next_cursor, first_log=None))
        idx += 1
        cursor = next_cursor
    return intervals


# Single-pass interval counting + first-log capture for performance.
def compute_interval_stats(intervals: List[IntervalRow], entries: List[LogEntry]) -> Tuple[List[int], List[Optional[LogEntry]]]:
    if not intervals:
        return [], []
    counts = [0 for _ in intervals]
    first_logs: List[Optional[LogEntry]] = [None for _ in intervals]
    i = 0
    for entry in sorted(entries, key=lambda e: e.ts):
        while i < len(intervals) and entry.ts >= intervals[i].end:
            i += 1
        if i >= len(intervals):
            break
        if intervals[i].start <= entry.ts < intervals[i].end:
            counts[i] += 1
            if first_logs[i] is None:
                first_logs[i] = entry
    return counts, first_logs


# Attach the first log to each interval row.
def assign_first_logs(intervals: List[IntervalRow], first_logs: List[Optional[LogEntry]]) -> List[IntervalRow]:
    updated: List[IntervalRow] = []
    for interval, first in zip(intervals, first_logs):
        updated.append(IntervalRow(index=interval.index, start=interval.start, end=interval.end, first_log=first))
    return updated


# Extract all entries for a given interval window.
def entries_in_interval(entries: List[LogEntry], interval: IntervalRow) -> List[LogEntry]:
    return [e for e in entries if interval.start <= e.ts < interval.end]


# Convert datetime to date for date inputs.
def coerce_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


# Safely extract a date from nested date input structures.
def extract_date(value) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (list, tuple)) and value:
        return extract_date(value[0])
    return None


# Normalize the date input into (start_date, end_date).
def normalize_date_range(value) -> Tuple[date, date]:
    if isinstance(value, (list, tuple)) and value:
        start = extract_date(value)
        end = extract_date(value[-1])
        if start and end:
            return start, end
    return coerce_date(value), coerce_date(value)


# Streamlit page layout.
st.set_page_config(page_title="Syslog Interval Viewer", layout="wide")

# File + console logging for debugging.
logger = logging.getLogger("syslog_viewer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("syslog_viewer.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Global UI tweaks (dialog sizing, scrollbar styling, etc.).
st.markdown(
    """
    <style>
    div.stButton > button { width: 100%; }
    div[role="dialog"] { width: 100vw !important; max-width: 4000px !important; }
    div[role="dialog"] > div { width: 98vw !important; max-width: 4000px !important; }
    div[role="dialog"] > div > div { resize: both; overflow: auto; max-height: 90vh; scrollbar-gutter: stable; }
    div[data-testid="stCodeBlock"] { max-height: 70vh; overflow-y: scroll; scrollbar-gutter: stable; }
    div[data-testid="stCodeBlock"] pre { overflow-x: auto; white-space: pre; }
    ::-webkit-scrollbar { width: 14px; height: 14px; }
    ::-webkit-scrollbar-thumb { background: #888; border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cache fetched logs across runs to avoid re-querying journalctl.
if "log_cache" not in st.session_state:
    st.session_state.log_cache = {
        "entries": [],
        "ranges": [],
        "cursors": set(),
    }
    logger.info("Initialized log cache")

# Top-level UI title.
st.title("Syslog Interval Viewer")

# Sidebar controls.
with st.sidebar:
    st.header("Query")
    today = datetime.now().date()
    # Default date range is yesterday -> today.
    if "date_range_input" not in st.session_state:
        st.session_state["date_range_input"] = (today - timedelta(days=1), today)
    date_range = st.date_input("Date range", key="date_range_input")
    cols = st.columns(2)
    # Quick range handlers.
    def set_date_range(value):
        st.session_state["date_range_input"] = value

    yday = today - timedelta(days=1)
    cols[0].button("Yesterday", on_click=set_date_range, args=((yday, yday),))
    cols[1].button("Today", on_click=set_date_range, args=((today, today),))

    # 1-second granularity time selection.
    start_seconds = st.select_slider(
        "Start time",
        options=range(0, 86400),
        value=0,
        format_func=lambda s: seconds_to_time(s).strftime("%I:%M:%S %p"),
    )
    end_seconds = st.select_slider(
        "End time",
        options=range(0, 86400),
        value=86399,
        format_func=lambda s: seconds_to_time(s).strftime("%I:%M:%S %p"),
    )
    start_time = seconds_to_time(start_seconds)
    end_time = seconds_to_time(end_seconds)
    # Interval length in seconds (default 10).
    interval_seconds = st.number_input("Interval seconds", min_value=1, value=10, step=1)
    fetch_clicked = st.button("Fetch logs")

# Normalize date range and build timezone-aware datetimes.
start_date, end_date = normalize_date_range(date_range)

start_dt = datetime.combine(start_date, start_time, tzinfo=local_tz())
end_dt = datetime.combine(end_date, end_time, tzinfo=local_tz())

# Validate range.
if end_dt <= start_dt:
    st.error("End time must be after start time.")
    st.stop()

# Local reference to cache for convenience.
cache = st.session_state.log_cache

# Fetch missing log ranges from journalctl.
if fetch_clicked:
    requested = (start_dt, end_dt)
    missing = subtract_ranges(requested, cache["ranges"])
    logger.info("Fetch requested: %s -> %s", format_dt(start_dt), format_dt(end_dt))
    logger.info("Missing ranges: %s", [(format_dt(s), format_dt(e)) for s, e in missing])
    new_entries: List[LogEntry] = []
    for miss_start, miss_end in missing:
        try:
            new_entries.extend(run_journalctl(miss_start, miss_end))
        except Exception as exc:
            logger.exception("Failed to fetch logs")
            st.error(f"Failed to fetch logs: {exc}")
            st.stop()
    cache["entries"] = dedupe_entries(new_entries, cache["entries"], cache["cursors"])
    cache["ranges"] = merge_ranges(cache["ranges"] + missing)
    logger.info("Cache now has %d entries", len(cache["entries"]))

# Pull relevant cached entries for current range.
entries_in_range = filter_entries(cache["entries"], start_dt, end_dt)

# Show warnings/errors for empty or partial ranges.
if fetch_clicked:
    if not entries_in_range:
        logger.warning("No logs found for requested range")
        st.error("No logs found for the requested range.")
        st.stop()
    min_ts = min(e.ts for e in entries_in_range)
    max_ts = max(e.ts for e in entries_in_range)
    if min_ts > start_dt or max_ts < end_dt:
        logger.warning("Partial availability for requested range")
        st.warning("Only part of the requested range is available in the journal.")

# Build intervals and compute counts + first log per interval.
intervals_base = build_intervals(start_dt, end_dt, int(interval_seconds))
counts, first_logs = compute_interval_stats(intervals_base, entries_in_range)
intervals = assign_first_logs(intervals_base, first_logs)

# Assemble rows for the interval table UI.
rows = []
last_date_label: Optional[str] = None
# Auto-focus table on the first interval with log activity.
first_nonzero_idx = next((i for i, c in enumerate(counts) if c > 0), None)
display_start = first_nonzero_idx if first_nonzero_idx is not None else 0
display_intervals = intervals[display_start:]
display_counts = counts[display_start:]

# Format interval display rows.
for interval, count in zip(display_intervals, display_counts):
    first_log_raw = interval.first_log.message if interval.first_log else "(no logs)"
    first_log = sanitize_text(first_log_raw, max_len=200)
    label, last_date_label = format_interval_label(interval, last_date_label)
    rows.append({
        "Interval": label,
        "Log lines": count,
        "First log": first_log,
        "View": "🔍",
        "_interval": interval,
    })

# Table view header.
st.subheader("Intervals")

if rows:
    # Build editor dataframe with a View checkbox column.
        df = pd.DataFrame(rows)
        max_count = max(display_counts) if display_counts else 0

        df["View"] = False
        df = df[["View", "Interval", "Log lines", "First log", "_interval"]]
        # Changing the widget key resets checkbox state.
        if "interval_table_key" not in st.session_state:
            st.session_state["interval_table_key"] = 0
        editor_key = f"interval_table_{st.session_state['interval_table_key']}"

        edited_df = st.data_editor(
            df.drop(columns=["_interval"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Log lines": st.column_config.ProgressColumn(
                    "Log lines",
                    min_value=0,
                    max_value=max_count if max_count > 0 else 1,
                    format="%d",
                ),
                "View": st.column_config.CheckboxColumn("🔍", width="small"),
            },
            disabled=["Interval", "Log lines", "First log"],
            height=600,
            key=editor_key,
        )

        # Open modal for the first selected row.
        selected_indices = edited_df.index[edited_df["View"] == True].tolist()
        if selected_indices:
            selected_row = rows[selected_indices[0]]
            selected_interval = selected_row["_interval"]
            interval_logs = entries_in_interval(entries_in_range, selected_interval)
            log_text = "\n".join(sanitize_text(f"{format_dt(e.ts)} | {e.message}") for e in interval_logs)

            # Modal dialog when available, expander fallback otherwise.
            if hasattr(st, "dialog"):
                @st.dialog("Interval details")
                def _show_interval_details() -> None:
                    st.write(f"Interval: {format_dt(selected_interval.start)} → {format_dt(selected_interval.end)}")
                    if not log_text:
                        st.info("No logs in this interval.")
                    else:
                        st.markdown(
                            """
                            <style>
                            .log-scroll {
                                max-height: 70vh;
                                overflow-y: scroll;
                                overflow-x: auto;
                                scrollbar-gutter: stable;
                                border: 1px solid rgba(255,255,255,0.1);
                                padding: 8px;
                                white-space: pre;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"<div class='log-scroll'>{escape(log_text)}</div>", unsafe_allow_html=True)
                    # Close button clears the checkbox via key increment.
                    if st.button("Close", type="primary"):
                        st.session_state["interval_table_key"] += 1
                        st.rerun()

                _show_interval_details()
            else:
                with st.expander("Interval details", expanded=True):
                    st.write(f"Interval: {format_dt(selected_interval.start)} → {format_dt(selected_interval.end)}")
                    if not log_text:
                        st.info("No logs in this interval.")
                    else:
                        st.code(log_text, language="text")
else:
    # No intervals to display for the chosen range.
        st.info("No intervals to display.")
