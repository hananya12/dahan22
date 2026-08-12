"""
reports.py
----------
Automatic Retail Analytics reports for Corewise (Task 4).

Reports are derived ENTIRELY from data the dashboard already persists to
``data/history/<store_id>.json`` (see app.py's _save_day_record): each
finished day is stored with hourly TAM/SOM buckets, peak-inside per hour,
totals, average stay time and weekday. Nothing here is fabricated — a
report only ever summarizes real, previously observed days.

Generation is automatic and idempotent:
  * ``ensure_reports_up_to_date(store_id)`` is called every time the
    Reports page renders. It looks at which complete days/weeks/months
    exist in history but do NOT yet have a saved report, and generates +
    saves exactly those. Re-running it is a no-op once everything is
    current, so it is cheap to call on every rerun.

Saved reports live under ``data/reports/<store_id>/<period>/<key>.json``
so previous reports can always be viewed later. The JSON payload is
structured (not pre-rendered text) to keep it ready for future export
formats (PDF / CSV / email) without regeneration.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPORTS_DIR = Path(__file__).resolve().parent / "data" / "reports"
HISTORY_DIR = Path(__file__).resolve().parent / "data" / "history"

PERIOD_DAILY = "daily"
PERIOD_WEEKLY = "weekly"
PERIOD_MONTHLY = "monthly"
PERIODS = (PERIOD_DAILY, PERIOD_WEEKLY, PERIOD_MONTHLY)


# ---------------------------------------------------------------------------
# History access
# ---------------------------------------------------------------------------

def _load_history(store_id: str) -> Dict[str, Dict[str, Any]]:
    path = HISTORY_DIR / f"{store_id}.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_day(day_str: str) -> Optional[date]:
    try:
        return datetime.strptime(day_str, "%Y-%m-%d").date()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _empty_metrics() -> Dict[str, Any]:
    return {
        "total_visitors": 0,
        "tam": 0,
        "sam": 0,
        "som": 0,
        "conversion_rate": 0.0,
        "avg_stay_time": None,
        "hourly_tam": {str(h): 0 for h in range(24)},
        "hourly_som": {str(h): 0 for h in range(24)},
        "peak_inside": {str(h): 0 for h in range(24)},
        "busy_hours": [],
        "days_counted": 0,
        "events": [],
    }


def _fold_day(agg: Dict[str, Any], record: Dict[str, Any]) -> None:
    """Fold one persisted day record into an aggregate metrics dict."""
    tam = int(record.get("total_tam", 0) or 0)
    som = int(record.get("total_som", 0) or 0)
    agg["tam"] += tam
    agg["som"] += som
    # SAM isn't stored per-day historically; when absent it is treated as
    # 0 for the total (the live SAM metric remains available on the
    # dashboard). This keeps reports honest rather than inventing a value.
    agg["sam"] += int(record.get("total_sam", 0) or 0)
    agg["days_counted"] += 1

    for hour, val in (record.get("hourly_tam") or {}).items():
        agg["hourly_tam"][str(hour)] = agg["hourly_tam"].get(str(hour), 0) + int(val or 0)
    for hour, val in (record.get("hourly_som") or {}).items():
        agg["hourly_som"][str(hour)] = agg["hourly_som"].get(str(hour), 0) + int(val or 0)
    for hour, val in (record.get("peak_inside") or {}).items():
        prev = agg["peak_inside"].get(str(hour), 0)
        agg["peak_inside"][str(hour)] = max(prev, int(val or 0))

    stay = record.get("avg_stay_time")
    if stay is not None:
        agg.setdefault("_stay_samples", []).append(float(stay))


def _finalize(agg: Dict[str, Any]) -> Dict[str, Any]:
    agg["total_visitors"] = agg["tam"]
    agg["conversion_rate"] = round((agg["som"] / agg["tam"] * 100.0), 1) if agg["tam"] else 0.0

    stay_samples = agg.pop("_stay_samples", [])
    agg["avg_stay_time"] = round(sum(stay_samples) / len(stay_samples), 1) if stay_samples else None

    # Busy hours = the top hours by new-TAM (fall back to peak-inside).
    hourly = agg["hourly_tam"]
    if not any(hourly.values()):
        hourly = agg["peak_inside"]
    busy = sorted(
        ((int(h), v) for h, v in hourly.items() if v),
        key=lambda pair: (-pair[1], pair[0]),
    )[:3]
    agg["busy_hours"] = [{"hour": h, "value": v} for h, v in busy]
    return agg


def _aggregate_days(store_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = _empty_metrics()
    for rec in records:
        _fold_day(agg, rec)
    return _finalize(agg)


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def _week_key(d: date) -> str:
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _month_key(d: date) -> str:
    return f"{d.year}-{d.month:02d}"


def _period_bounds(period: str, key: str) -> Tuple[Optional[date], Optional[date]]:
    """Inclusive (start, end) dates a report key covers, for labeling."""
    try:
        if period == PERIOD_DAILY:
            d = _parse_day(key)
            return (d, d)
        if period == PERIOD_WEEKLY:
            iso_year, iso_week = int(key[:4]), int(key[6:])
            start = date.fromisocalendar(iso_year, iso_week, 1)
            return (start, start + timedelta(days=6))
        if period == PERIOD_MONTHLY:
            year, month = int(key[:4]), int(key[5:7])
            start = date(year, month, 1)
            end = date(year + (month // 12), (month % 12) + 1, 1) - timedelta(days=1)
            return (start, end)
    except Exception:
        pass
    return (None, None)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _report_path(store_id: str, period: str, key: str) -> Path:
    return REPORTS_DIR / store_id / period / f"{key}.json"


def _save_report(store_id: str, period: str, key: str, payload: Dict[str, Any]) -> None:
    path = _report_path(store_id, period, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_report(store_id: str, period: str, key: str) -> Optional[Dict[str, Any]]:
    path = _report_path(store_id, period, key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def list_reports(store_id: str, period: str) -> List[str]:
    """Report keys for a store/period, newest first."""
    folder = REPORTS_DIR / store_id / period
    if not folder.is_dir():
        return []
    keys = [p.stem for p in folder.glob("*.json")]
    return sorted(keys, reverse=True)


# ---------------------------------------------------------------------------
# Automatic generation
# ---------------------------------------------------------------------------

def _build_report(store_id: str, period: str, key: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = _aggregate_days(store_id, records)
    start, end = _period_bounds(period, key)
    return {
        "store_id": store_id,
        "period": period,
        "key": key,
        "range": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }


def _group_history(history: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Bucket every complete day record by daily/weekly/monthly key."""
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {p: {} for p in PERIODS}
    for day_str, rec in history.items():
        d = _parse_day(day_str)
        if d is None:
            continue
        grouped[PERIOD_DAILY].setdefault(day_str, []).append(rec)
        grouped[PERIOD_WEEKLY].setdefault(_week_key(d), []).append(rec)
        grouped[PERIOD_MONTHLY].setdefault(_month_key(d), []).append(rec)
    return grouped


def _current_period_keys(today: date) -> Dict[str, str]:
    """Keys for the still-in-progress periods, which must NOT be finalized
    yet (they'd be incomplete). Excluded from automatic generation."""
    return {
        PERIOD_DAILY: today.isoformat(),
        PERIOD_WEEKLY: _week_key(today),
        PERIOD_MONTHLY: _month_key(today),
    }


def ensure_reports_up_to_date(store_id: str, today: Optional[date] = None) -> int:
    """Generate + save any missing reports for completed periods.

    Idempotent: only creates reports that don't already exist and that
    cover a fully-elapsed period. Returns the number of new reports written.
    Safe to call on every page render.
    """
    today = today or date.today()
    history = _load_history(store_id)
    if not history:
        return 0

    grouped = _group_history(history)
    in_progress = _current_period_keys(today)
    created = 0

    for period in PERIODS:
        for key, records in grouped[period].items():
            if key == in_progress[period]:
                continue  # period still open — don't finalize it yet
            if load_report(store_id, period, key) is not None:
                continue  # already generated
            payload = _build_report(store_id, period, key, records)
            _save_report(store_id, period, key, payload)
            created += 1
    return created


def generate_report_now(store_id: str, period: str, key: str) -> Optional[Dict[str, Any]]:
    """Force-(re)build a single report for the given period/key, even if the
    period is still in progress. Used by an explicit "generate now" action so
    a store can see a partial current-period report on demand."""
    history = _load_history(store_id)
    if not history:
        return None
    grouped = _group_history(history)
    records = grouped.get(period, {}).get(key)
    if not records:
        return None
    payload = _build_report(store_id, period, key, records)
    _save_report(store_id, period, key, payload)
    return payload


def latest_available_keys(store_id: str) -> Dict[str, Optional[str]]:
    """Newest saved report key per period (or None), for default selection."""
    return {period: (list_reports(store_id, period)[:1] or [None])[0] for period in PERIODS}


# ---------------------------------------------------------------------------
# Export scaffolding (kept ready for future formats — Task 4 requirement)
# ---------------------------------------------------------------------------

def report_to_csv(report: Dict[str, Any]) -> str:
    """Flatten a report's hourly TAM/SOM into CSV text.

    Provided now so a future "Export" button has a real, working
    serializer to call; the Reports UI can wire a download to this without
    any change to generation or storage.
    """
    metrics = report.get("metrics", {})
    hourly_tam = metrics.get("hourly_tam", {})
    hourly_som = metrics.get("hourly_som", {})
    peak = metrics.get("peak_inside", {})
    lines = ["hour,new_tam,new_som,peak_inside"]
    for h in range(24):
        lines.append(
            f"{h:02d},{int(hourly_tam.get(str(h), 0))},"
            f"{int(hourly_som.get(str(h), 0))},{int(peak.get(str(h), 0))}"
        )
    return "\n".join(lines) + "\n"
