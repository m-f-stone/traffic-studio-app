# studio/analyzer.py
from pathlib import Path
import csv, re
from .roi_utils import normalize_for_tracking
from .logutil import setup_logger
log = setup_logger()

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

    with tracks_csv.open("r", newline="", encoding="utf-8") as f, \
         out_tracks.open("w", newline="", encoding="utf-8") as g:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames) + ["movement_ltr", "approach"]
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        for row in r:
            entry = row.get("entry_roi","") or ""
            exitr = row.get("exit_roi","") or ""
            approach, turn = _ltr_from(entry, exitr)
            ltr = f"{approach or 'NA'}-{turn or 'Unknown'}"
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
                  out_dir: Path, run_name="analysis", tracker="strongsort.yaml",
                  imgsz=640, conf=0.30, iou=0.50, device=0,
                  include_lane_rois: bool=False, lane_pattern: str=r"(lane|_ln|\b[ltr]\b|through|left|right)"):
    """
    Normalizes ROI JSON (template → flat), optionally forcing lane-ROIs 'include'=True
    for names matching `lane_pattern`, then runs your analyzer. Finally, post-process
    per-track results to add approach/L/T/R labels and an aggregate CSV.
    """
    from third_party.traffic_analysis_tracking import run_analysis

    out_dir.mkdir(parents=True, exist_ok=True)
    # Normalize ROI JSON for tracking script
    flat_json = out_dir / f"{run_name}_rois_flat.json"
    normalize_for_tracking(roi_json_template, flat_json,
                           include_lanes=include_lane_rois,
                           lane_pattern=lane_pattern)

    # Run tracking analysis via direct-call API (thread-safe)
    run_analysis(
        input_video=str(video),
        model=str(model_path),
        output=str(out_dir),
        run=run_name,
        rois_path=str(flat_json),
        tracker=tracker,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device
    )

    # Post-process to add L/T/R
    _postprocess_ltr(out_dir, run_name)
