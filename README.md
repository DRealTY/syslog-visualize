# Syslog Interval Viewer

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

## Notes
- Uses `journalctl` on Ubuntu to fetch logs for the selected date/time range.
- Row selection in the interval table requires Streamlit 1.33+.
