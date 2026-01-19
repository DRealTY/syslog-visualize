# Agent Notes — Syslog Interval Viewer

This document is for automated coding agents working on this repo. It summarizes architecture, workflow, and gotchas discovered during development.

## Project Overview
- Streamlit app that queries `journalctl` for a date/time range, chunks logs into fixed intervals, and displays the first log + count for each interval.
- Detail view shows all logs for the selected interval in a modal (or expander fallback).
- Logs are cached in-session to avoid re-fetching ranges.

## Key Files
- [app.py](app.py) — main Streamlit app.
- [run.sh](run.sh) — wrapper that starts Streamlit and opens the browser.
- [syslog_viewer.log](syslog_viewer.log) — runtime logging for debugging.
- [README.md](README.md) — user documentation.

## UI/UX Flow
1. Sidebar: date range, today/yesterday quick buttons, time sliders, interval seconds, fetch.
2. Main: interval table with progress-style log count and a **View** checkbox per row.
3. Clicking **View** opens a dialog with full logs for that interval.

## Known Gotchas / Quirks
### 1) Streamlit session state rules
- Do **not** set widget values in `st.session_state` after instantiation.
- For `st.data_editor`, avoid assigning to `st.session_state[key]` directly.
- Resetting the View checkbox is done by changing the widget key (`interval_table_key`) and re-running.

### 2) Dialog sizing / CSS selectors
- Streamlit dialogs are hard to size reliably; CSS selectors target `div[role="dialog"]`.
- Width is forced to near full-screen with `!important`.
- Sticky close button is not guaranteed; a dedicated **Close** button is included in the dialog.

### 3) Log rendering
- Log content can include control characters or non-ASCII bytes. `sanitize_text()` strips non-printable chars.
- Modal content uses a custom `<div>` with `white-space: pre` to preserve line breaks.
- Horizontal scrolling is enabled to prevent truncation.

### 4) Performance
- Interval counting uses a single pass (`compute_interval_stats`) instead of per-interval filtering.
- Interval list auto-focuses on the first non-zero interval for quick navigation.

### 5) Permissions
- `journalctl` can require elevated permissions or membership in `systemd-journal` group.

## How the Modal Works
- `st.data_editor` is used with a **View** checkbox column.
- When a checkbox is selected, a dialog shows the interval details.
- After closing, the table key is incremented to clear the checkbox.

## Tips for Future Work
- If you need richer styling for the interval table, consider a custom component (Streamlit Components) instead of raw HTML injection.
- Avoid embedding raw log lines directly into HTML attributes due to encoding and size limits.
- Keep dialog content inside a fixed-height container to preserve scrollbars.

## Running / Debugging
```bash
./run.sh
# or
streamlit run app.py
```

Check `syslog_viewer.log` for errors or fetch ranges.
