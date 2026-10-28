# studio/analyzer.py
"""Traffic analysis entry point: flatten the ROI template, run tracking, and
post-process per-track results into approach / left-through-right labels."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, Optional

from .logutil import setup_logger
from .roi_utils import normalize_for_tracking

log = setup_logger()

DEFAULT_TRACKER = "bytetrack.yaml"   # ultralytics ships bytetrack.yaml and botsort.yaml
DEFAULT_LANE_PATTERN = r"(lane|_ln|\b[ltr]\b|through|left|right)"

# --- helpers to read cardinal approach from ROI names ---
_CARDINALS = {
    "north": "N", "south": "S", "east": "E", "west": "W",
    "nb": "N", "sb": "S", "eb": "E", "wb": "W"
}

def _detect_approach(name: str) -> str | None:
    s = name.lower()
    # common abbreviations: NB, SB, EB, WB
    m = re.search(r"\b([nsew])b\b", s)
    if m: return m.group(1).upper()
    # full tokens north/south/east/west anywhere
    for k, v in _CARDINALS.items():
        if re.search(rf"\b{k}\b", s):
            return v
    # prefix-style 'north_' or 'n-'
    m = re.match(r"\s*([nsew])([\W_]|$)", s)
    if m: return m.group(1).upper()
    # embedded 'northlane', 'west_left' etc.
    for k, v in _CARDINALS.items():
        if k in s:
            return v
    return None

# N→E right, N→W left, N→S through; etc.
_TURN = {
    ("N","E"): "Right", ("N","W"): "Left",  ("N","S"): "Through",
    ("E","S"): "Right", ("E","N"): "Left",  ("E","W"): "Through",
    ("S","W"): "Right", ("S","E"): "Left",  ("S","N"): "Through",
    ("W","N"): "Right", ("W","S"): "Left",  ("W","E"): "Through",
}

def _ltr_from(entry_roi: str, exit_roi: str) -> tuple[str,str] | tuple[None,None]:
    a = _detect_approach(entry_roi)
    b = _detect_approach(exit_roi)
    if not a or not b: return None, None
    if a == b: return a, "U/Loop"
    return a, _TURN.get((a,b), "Unknown")

def _postprocess_ltr(out_dir: Path, run_name: str):
    tracks_csv = out_dir / f"{run_name}_tracks.csv"
    if not tracks_csv.exists():
        log.warning("LTR post-process: tracks CSV not found: %s", tracks_csv)
        return

    # 1) write per-track file with extra movement_ltr column
    out_tracks = out_dir / f"{run_name}_tracks_ltr.csv"
    agg = {}  # (approach, turn, class) -> count

    with tracks_csv.open("r", newline="", encoding="utf-8-sig") as f, \
         out_tracks.open("w", newline="", encoding="utf-8") as g:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames) + ["movement_ltr", "approach"]
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        for row in r:
            entry = row.get("entry_roi","") or ""
            exitr = row.get("exit_roi","") or ""
            approach, turn = _ltr_from(entry, exitr)
            row["approach"] = approach or ""
            row["movement_ltr"] = turn or "Unknown"
            w.writerow(row)
            # aggregate by class and LTR
            cls = row.get("class","")
            if approach and turn:
                agg[(approach, turn, cls)] = agg.get((approach, turn, cls), 0) + 1

    # 2) write aggregate movements_ltr.csv
    out_mov = out_dir / f"{run_name}_movements_ltr.csv"
    with out_mov.open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["approach","turn","class","count"])
        for (app, turn, cls), n in sorted(agg.items()):
            w.writerow([app, turn, cls, n])

    log.info("Wrote LTR files:\n  %s\n  %s", out_tracks, out_mov)


def analyze_video(video: Path, model_path: Path, roi_json_template: Path,
                  out_dir: Path, run_name="analysis", tracker=DEFAULT_TRACKER,
                  imgsz=640, conf=0.30, iou=0.50, device=None,
                  include_lane_rois: bool = False, lane_pattern: str = DEFAULT_LANE_PATTERN,
                  write_video: bool = True,
                  progress_callback: Optional[Callable[[float, str], None]] = None,
                  cancel_check: Optional[Callable[[], bool]] = None) -> dict:
    """
    Normalizes the ROI JSON (template -> flat), optionally forcing lane ROIs to
    include=True for names matching `lane_pattern`, runs tracking, then
    post-processes per-track results into approach / L-T-R labels plus an
    aggregate CSV.

    device None/'auto' picks CUDA when available, otherwise CPU. On cancel the
    tracks seen so far are still written and post-processed.

    Returns a dict with the output paths and a "cancelled" flag.
    """
    from third_party.traffic_analysis_tracking import run_analysis

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(0.0, "Preparing ROIs")
    flat_json = out_dir / f"{run_name}_rois_flat.json"
    normalize_for_tracking(Path(roi_json_template), flat_json,
                           include_lanes=include_lane_rois,
                           lane_pattern=lane_pattern)

    def _tracking_progress(frac: float, msg: str):
        if progress_callback:
            progress_callback(0.02 + 0.93 * frac, msg)

    out = run_analysis(
        input_video=str(video),
        model=str(model_path),
        output=str(out_dir),
        run=run_name,
        rois_path=str(flat_json),
        tracker=tracker,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        write_video=write_video,
        progress_callback=_tracking_progress,
        cancel_check=cancel_check,
    )

    if progress_callback:
        progress_callback(0.96, "Classifying movements (L/T/R)")
    _postprocess_ltr(out_dir, run_name)

    cancelled = bool(out.get("cancelled"))
    if progress_callback:
        progress_callback(1.0, "Analysis cancelled; partial results written" if cancelled else "Analysis complete")
    return {
        "tracks_csv": out["tracks_csv"],
        "movements_csv": out["movements_csv"],
        "tracks_ltr_csv": out_dir / f"{run_name}_tracks_ltr.csv",
        "movements_ltr_csv": out_dir / f"{run_name}_movements_ltr.csv",
        "video": out.get("video"),
        "frames": out.get("frames", 0),
        "cancelled": cancelled,
    }
