# Corewise V1 — AI Retail Analytics Platform

Real-time retail foot-traffic analytics from a live camera feed.
**Not a security system** — no recordings, no video storage, only
aggregated business metrics (TAM / SAM / SOM / conversion).

## Architecture

```
Dashboard (Streamlit, app.py)   ← N instances
        ⇅ WebSocket
Corewise Server (corewise_server.py)   ← in-memory state, no files
        ⇅ WebSocket
AI Engine (main.py)   ← 1 camera today, N cameras later
```

`corewise_control.py` is the shared client used by both `main.py`
(role `"engine"`) and `app.py` (role `"dashboard"`) to talk to
`corewise_server.py`. There is **no JSON/TXT/CSV file** used for live
synchronization — `corewise_state.json` from the original design has
been retired in favor of the in-memory `SystemState` on the server.
`zones.json` and `line_config.json` remain as **one-time setup data**
(the TAM polygon / entrance line), not live sync data.

## Running it

Three processes, in this order:

```bash
# 1. Start the central hub
python corewise_server.py

# 2. Start the AI engine (needs Iriun Webcam running)
python main.py

# 3. Start the dashboard
streamlit run app.py
```

Toggling **Blur Faces** or **Night Vision** in the dashboard reaches
`main.py` within one WebSocket round trip (typically single-digit
milliseconds on localhost) — no restart, no polling, no manual page
refresh.

## Files

| File | Responsibility |
|---|---|
| `corewise_server.py` | Central WebSocket hub — relays engine ⇄ dashboard, holds live state in memory |
| `corewise_control.py` | Background-thread WebSocket client shared by engine and dashboard |
| `main.py` | Camera → YOLOv8 + ByteTrack → tracking → effects → telemetry |
| `detection.py` | YOLO model wrapper, hot-swappable model/confidence |
| `tracking.py` | TAM / SAM / SOM / Inside / Avg Stay / Conversion logic |
| `effects.py` | Face Blur, Night Vision (CCTV-style), future effects |
| `face_recognition.py` | Optional named-person identification |
| `camera.py` | Iriun Webcam discovery + connection |
| `app.py` | Streamlit investor-grade dashboard |
| `setup_zones.py` / `setup_line.py` | One-time interactive setup tools |
| `zones.json` / `line_config.json` | Static setup output (not live sync) |

## Designed for growth

Adding multiple cameras, multiple stores, a REST API, a mobile app, or
a database does not require changing this architecture: they all
become additional consumers/producers on `corewise_server.py`'s
in-memory `SystemState`, tagged by a future `client_id` / `store_id`.
