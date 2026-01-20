# Syslog Interval Viewer

Streamlit UI for chunking Ubuntu `journalctl` logs into fixed time intervals and quickly inspecting each interval.

<p align="center">
  <img src="media/syslog-visualize.gif" width="900">
</p>

## Features
- Date/time range + interval size (seconds).
- Caches fetched logs and only requests missing ranges.
- Interval table shows first log + count with heat-style progress bar.
- One-click details modal with full logs for the selected interval.
- Auto-focuses on the first interval with non-zero logs.
- Debug logging to `syslog_viewer.log` in the project directory.

## Requirements
- Linux with `journalctl` available (systemd).
- Python 3.9+ recommended.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

Or use the wrapper to auto-open the browser:
```bash
./run.sh
```

## Usage
1. Pick a date range.
2. Use the time sliders to select start/end times (12‑hour clock).
3. Set interval seconds (default 10).
4. Click **Fetch logs**.
5. Use the **🔍** checkbox column to open interval details.

## Permissions
`journalctl` may require elevated permissions on some systems. If you get empty results or permission errors, run Streamlit with sufficient access or add your user to the `systemd-journal` group.

## Troubleshooting
- **No logs found**: the selected range may not exist in the journal or lacks permissions.
- **Partial availability warning**: the journal only has logs for part of the requested range.
- **Port already in use**: stop the existing Streamlit process or change `PORT` in `run.sh`.

## Development Notes
- The interval detail modal uses `st.dialog` if available; otherwise it falls back to an expander.
- The **View** column is implemented via `st.data_editor` to allow click-to-open without slow row selection.
- Log text is sanitized to prevent rendering issues from control characters.
- UI tweaks for dialog width/scrollbars are applied via CSS in `app.py`.
