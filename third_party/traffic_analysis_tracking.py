#!/usr/bin/env python3
"""
Traffic patterns with tracking, ROI paths, and speeds.

Streams YOLO detection + tracking over a video, accumulates a trajectory per
track id, records the sequence of ROIs each track passes through, and writes:

  <run>_tracks.csv      one row per track (entry/exit ROI, distance, speeds, stops)
  <run>_movements.csv   origin-destination counts by class
  <run>_annotated.mp4   annotated video (optional)

Usable from the command line or through run_analysis() (studio.analyzer and
traffic_studio.services.analyzer). The helpers above streaming_patterns() have
no torch dependency so they can be unit-tested without a model.
"""
import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon

log = logging.getLogger("traffic_studio.tracking")

CSV_ENCODING = "utf-8-sig"
# ultralytics ships exactly these two tracker configs; anything else must be a path to a YAML.
SUPPORTED_TRACKERS = ("bytetrack.yaml", "botsort.yaml")
DEFAULT_TRACKER = "bytetrack.yaml"
TRACK_TIMEOUT_FRAMES = 30   # finalize a track once it has been unseen for this many frames
TRAIL_LEN = 30              # breadcrumb points drawn per track
STOP_SPEED_MPS = 0.2        # samples slower than this (m/s) count as stopped; needs calibration
STOP_MIN_SAMPLES = 3        # consecutive stopped samples that count as one stop
ROI_PATH_SEP = " -> "
TRACK_COLUMNS = [
    "track_id", "class", "first_frame", "last_frame", "duration_s",
    "entry_roi", "exit_roi", "roi_path",
    "distance", "distance_units", "avg_speed", "max_speed", "speed_units",
    "stops",
]


# -------------------------
# ROI loading & calibration
# -------------------------
def load_rois(roi_path):
    """
    ROI JSON format:
    {
      "NS Roadway": [[x,y], ...],
      "EW Roadway": [[x,y], ...],
      ...
      // optional calibration block (either-or; both optional)
      "calibration": {
         "meters_per_pixel": 0.024   // simple scale
         // OR
         "image_points": [[x,y], ... 4+ pts],
         "world_points_m": [[X,Y], ... matching 4+ pts in meters]
      }
    }
    Returns ({name: shapely Polygon}, calibration dict). ROIs with fewer than
    three distinct points are skipped with a warning.
    """
    with open(roi_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    calib = raw.pop("calibration", None)

    def clean_pts(pts):
        uniq = []
        for x, y in pts:
            if not uniq or (uniq[-1][0] != x or uniq[-1][1] != y):
                uniq.append((x, y))
        if len(uniq) > 1 and uniq[0] == uniq[-1]:
            uniq.pop()
        return uniq

    rois = {}
    for name, pts in raw.items():
        if not isinstance(pts, list):
            continue
        cleaned = clean_pts(pts)
        if len(cleaned) < 3:
            log.warning("ROI %r has fewer than 3 distinct points; skipped.", name)
            continue
        rois[name] = Polygon(cleaned)
    return rois, calib or {}


def build_homography(calib):
    """Return (H, use_homography, meters_per_pixel) from calibration dict."""
    mpp = calib.get("meters_per_pixel", None)
    ip = calib.get("image_points", None)
    wp = calib.get("world_points_m", None)
    if ip and wp and len(ip) >= 4 and len(ip) == len(wp):
        src = np.array(ip, dtype=np.float32)
        dst = np.array(wp, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
        return H, True, None
    elif mpp:
        return None, False, float(mpp)
    else:
        return None, False, None


def px_to_world(pt_xy, H=None, meters_per_px=None):
    """Pixel (x,y) -> world (X,Y) in metres when calibrated; else pixels.
    Returns (X, Y, is_world)."""
    x, y = pt_xy
    if H is not None:
        p = np.array([[x, y, 1.0]], dtype=np.float32).T
        pw = H @ p
        pw = pw / (pw[2] + 1e-9)
        return float(pw[0]), float(pw[1]), True
    elif meters_per_px is not None:
        return x * meters_per_px, y * meters_per_px, True
    else:
        return float(x), float(y), False


def resolve_device(requested=None):
    """None/'auto' -> CUDA 0 when available else 'cpu'; 'cpu' -> 'cpu';
    an explicit CUDA device is passed through only when CUDA is available."""
    def _cuda():
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    req = "" if requested is None else str(requested).strip().lower()
    if req in ("", "auto"):
        return 0 if _cuda() else "cpu"
    if req == "cpu":
        return "cpu"
    if _cuda():
        return int(req) if req.isdigit() else requested
    log.warning("Requested device %r but CUDA is not available; using CPU.", requested)
    return "cpu"


# -------------------------
# Tracking & feature logic
# -------------------------
class TrackState:
    __slots__ = (
        "cls", "name", "first_frame", "last_frame",
        "times", "px", "py", "rois", "max_conf", "trail"
    )

    def __init__(self, cls_id, class_name, frame_idx, conf):
        self.cls = int(cls_id)
        self.name = class_name
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.times = []  # t_sec per sample
        self.px = []     # x in world or pixel units
        self.py = []     # y in world or pixel units
        self.rois = []   # ROI name sequence (dedup)
        self.max_conf = float(conf)
        self.trail = deque(maxlen=TRAIL_LEN)  # pixel breadcrumbs for the video


def compress_seq(seq):
    out = []
    prev = None
    for s in seq:
        if s != prev:
            out.append(s)
            prev = s
    return out


def movement_label(path_seq):
    """Simple movement label from an ROI path sequence. ASCII only: it is
    written to CSV on Windows and rendered with OpenCV's Hershey fonts."""
    if not path_seq:
        return "unknown"
    if len(path_seq) == 1:
        return f"within {path_seq[0]}"
    if path_seq[0] == path_seq[-1]:
        return f"u/loop in {path_seq[0]}"
    return f"{path_seq[0]} -> {path_seq[-1]}"


def compute_speeds(xs, ys, ts, stop_speed=None, stop_min_samples=STOP_MIN_SAMPLES):
    """Per-sample speeds (units of xs/ys per second), their mean and max, and
    the number of stops. Stops are only counted when stop_speed is given, i.e.
    when the coordinates are metric; a m/s threshold is meaningless in px/s."""
    if len(ts) < 2:
        return [], 0.0, 0.0, 0
    speeds = []
    stops = 0
    stop_run = 0
    for i in range(1, len(ts)):
        dt = max(ts[i] - ts[i - 1], 1e-9)
        v = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) / dt
        speeds.append(v)
        if stop_speed is None:
            continue
        if v < stop_speed:
            stop_run += 1
        else:
            if stop_run >= stop_min_samples:
                stops += 1
            stop_run = 0
    if stop_speed is not None and stop_run >= stop_min_samples:
        stops += 1
    return speeds, float(np.mean(speeds)), float(np.max(speeds)), stops


def path_length(xs, ys):
    return sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs)))


def finalize_track(tid, st, fps, world_mode, stop_speed_mps=STOP_SPEED_MPS):
    """Build the tracks-CSV row for a finished track; None if it has < 2 samples."""
    if st is None or len(st.times) < 2:
        return None
    _, avg_v, max_v, stops = compute_speeds(
        st.px, st.py, st.times, stop_speed=stop_speed_mps if world_mode else None)
    dist = path_length(st.px, st.py)
    duration = (st.last_frame - st.first_frame + 1) / fps
    path_seq = compress_seq(st.rois)
    return [
        tid, st.name, st.first_frame, st.last_frame, round(duration, 3),
        path_seq[0] if path_seq else "", path_seq[-1] if path_seq else "", ROI_PATH_SEP.join(path_seq),
        round(dist, 3), "m" if world_mode else "px",
        round(avg_v, 3), round(max_v, 3), "m/s" if world_mode else "px/s",
        stops,
    ]


# -------------------------
# Annotated video output
# -------------------------
class AnnotatedVideoWriter:
    """Writes BGR frames to an .mp4.

    Prefers piping raw frames to ffmpeg/libx264, which is multi-threaded and
    several times faster than OpenCV's single-threaded mp4v encoder at full
    resolution. Falls back to OpenCV when ffmpeg is missing or fails to start.
    """

    def __init__(self, path, w, h, fps, prefer_ffmpeg=True):
        self.path = Path(path)
        self.w, self.h, self.fps = int(w), int(h), float(fps)
        self._proc = None
        self._cv = None
        self.frames = 0
        ffmpeg = shutil.which("ffmpeg") if prefer_ffmpeg else None
        if ffmpeg:
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.w}x{self.h}",
                "-r", f"{self.fps:.6f}", "-i", "-",
                "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
            ]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as e:
                log.warning("Could not start ffmpeg encoder (%s); using OpenCV writer.", e)
                self._proc = None
        if self._proc is None:
            self._open_cv()

    def _open_cv(self):
        self._cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.w, self.h))

    @property
    def backend(self):
        return "ffmpeg/libx264" if self._proc is not None else "opencv/mp4v"

    def write(self, frame):
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
                self.frames += 1
                return
            except (BrokenPipeError, OSError):
                rc = self._proc.poll()
                log.warning("ffmpeg encoder stopped (exit %s); switching to OpenCV writer. "
                            "Frames written so far are lost.", rc)
                self._proc = None
                self._open_cv()
        self._cv.write(frame)
        self.frames += 1

    def release(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.wait()
            self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None


# -------------------------
# Main streaming loop
# -------------------------
def streaming_patterns(
    model_path, input_video, output_dir, run_name,
    tracker_cfg=DEFAULT_TRACKER, imgsz=640,
    roi_polygons=None, calib=None, conf=0.3, iou=0.5, device=None, *,
    write_video=True,
    track_timeout_frames=TRACK_TIMEOUT_FRAMES,
    stop_speed_mps=STOP_SPEED_MPS,
    progress_callback=None,
    cancel_check=None,
):
    """
    Stream detection+tracking, build trajectories & behaviour features, and
    save per-track CSV, movement counts and (optionally) an annotated video.

    progress_callback(fraction, message) is called roughly every 0.5 % of the
    video; cancel_check() is polled every frame and, when true, the tracks seen
    so far are finalized and written before returning.

    Returns {"tracks_csv", "movements_csv", "video", "frames", "cancelled"}.
    """
    from ultralytics import YOLO  # lazy: keeps the helpers above importable without torch

    if tracker_cfg not in SUPPORTED_TRACKERS and not Path(tracker_cfg).is_file():
        raise ValueError(
            f"Unknown tracker {tracker_cfg!r}. Use one of {SUPPORTED_TRACKERS} or a path to a tracker YAML.")

    dev = resolve_device(device)
    model = YOLO(str(model_path))
    class_names = model.names

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv = out_dir / f"{run_name}_tracks.csv"
    moves_csv = out_dir / f"{run_name}_movements.csv"
    video_out = out_dir / f"{run_name}_annotated.mp4" if write_video else None

    # Calibration
    H, use_H, mpp = build_homography(calib or {})
    world_mode = use_H or (mpp is not None)
    stop_speed = stop_speed_mps if world_mode else None

    # Video info
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    roi_polygons = roi_polygons or {}

    def roi_name_pixel(cx, cy):
        for rn, poly in roi_polygons.items():
            if bool(shapely.contains_xy(poly, cx, cy)):
                return rn
        return None

    active = {}      # tid -> TrackState
    last_seen = {}   # tid -> frame_idx
    writer = None
    cancelled = False
    frames_done = 0
    report_every = max(1, total_frames // 200) if total_frames else 30

    with tracks_csv.open("w", newline="", encoding=CSV_ENCODING) as ftrk:
        tw = csv.writer(ftrk)
        tw.writerow(TRACK_COLUMNS)

        def finish(tid):
            st = active.pop(tid, None)
            last_seen.pop(tid, None)
            row = finalize_track(tid, st, fps, world_mode, stop_speed)
            if row:
                tw.writerow(row)

        stream = model.track(
            source=str(input_video), stream=True, imgsz=imgsz, tracker=tracker_cfg,
            conf=conf, iou=iou, persist=True, device=dev, verbose=False,
        )
        for frame_idx, result in enumerate(stream):
            if cancel_check and cancel_check():
                cancelled = True
                log.warning("Analysis cancelled at frame %d; writing partial results.", frame_idx)
                break

            frame_time = frame_idx / fps
            # One copy of the original frame; drawing our own labels replaces
            # result.plot() + a second pass of drawing on top of it.
            frame = result.orig_img.copy() if write_video else None
            if frame is not None and writer is None:
                h, w = result.orig_shape[:2]
                writer = AnnotatedVideoWriter(video_out, w, h, fps)
                log.info("Annotated video -> %s (%s)", video_out.name, writer.backend)

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                ids = boxes.id.int().tolist()
                clss = boxes.cls.int().tolist()
                confs = boxes.conf.tolist()
                xyxy = boxes.xyxy.tolist()
                for tid, cls, confd, (x1, y1, x2, y2) in zip(ids, clss, confs, xyxy):
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                    X, Y, _ = px_to_world((cx, cy), H, mpp)

                    st = active.get(tid)
                    if st is None:
                        st = active[tid] = TrackState(cls, class_names.get(cls, str(cls)), frame_idx, confd)
                    st.last_frame = frame_idx
                    st.max_conf = max(st.max_conf, confd)
                    st.times.append(frame_time)
                    st.px.append(X)
                    st.py.append(Y)
                    st.trail.append((int(cx), int(cy)))
                    rn = roi_name_pixel(cx, cy)
                    if rn is not None and (not st.rois or st.rois[-1] != rn):
                        st.rois.append(rn)
                    last_seen[tid] = frame_idx

                    if frame is not None:
                        label = f"ID {tid} {st.name} {confd:.2f} | {movement_label(compress_seq(st.rois))}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x1), max(20, int(y1) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if frame is not None:
                for st in active.values():
                    if len(st.trail) >= 2:
                        cv2.polylines(frame, [np.array(st.trail, dtype=np.int32)], False, (0, 200, 255), 2)
                writer.write(frame)

            # Finalize tracks that vanished
            for tid in [t for t, lf in last_seen.items() if frame_idx - lf > track_timeout_frames]:
                finish(tid)

            frames_done = frame_idx + 1
            if progress_callback and frames_done % report_every == 0:
                frac = frames_done / total_frames if total_frames else 0.0
                progress_callback(min(frac, 0.999),
                                  f"Frame {frames_done}/{total_frames or '?'} - {len(active)} active tracks")

        # Flush any remaining active tracks at end-of-video (or on cancel)
        for tid in list(active.keys()):
            finish(tid)

    if writer is not None:
        writer.release()

    # Aggregate movements (simple O-D from ROI path)
    counts = defaultdict(lambda: defaultdict(int))  # movement -> class -> count
    with tracks_csv.open("r", newline="", encoding=CSV_ENCODING) as f:
        for row in csv.DictReader(f):
            seq = row["roi_path"].split(ROI_PATH_SEP) if row["roi_path"] else []
            counts[movement_label(seq)][row["class"]] += 1
    with moves_csv.open("w", newline="", encoding=CSV_ENCODING) as f:
        w = csv.writer(f)
        w.writerow(["movement", "class", "count"])
        for move, per_cls in counts.items():
            for cls, n in per_cls.items():
                w.writerow([move, cls, n])

    log.info("Tracks -> %s | Movements -> %s%s", tracks_csv, moves_csv,
             f" | Video -> {video_out}" if video_out else "")
    return {"tracks_csv": tracks_csv, "movements_csv": moves_csv, "video": video_out,
            "frames": frames_done, "cancelled": cancelled}


def run_analysis(input_video, model, output, run, rois_path,
                 tracker=DEFAULT_TRACKER, imgsz=640,
                 conf=0.30, iou=0.50, device=None, *,
                 write_video=True,
                 track_timeout_frames=TRACK_TIMEOUT_FRAMES,
                 stop_speed_mps=STOP_SPEED_MPS,
                 progress_callback=None, cancel_check=None):
    """Direct-call API for running traffic analysis (thread-safe, no sys.argv).

    Returns the dict from streaming_patterns().
    """
    rois, calib = load_rois(rois_path)
    return streaming_patterns(
        model_path=model,
        input_video=input_video,
        output_dir=output,
        run_name=run,
        tracker_cfg=tracker,
        imgsz=imgsz,
        roi_polygons=rois,
        calib=calib,
        conf=conf,
        iou=iou,
        device=device,
        write_video=write_video,
        track_timeout_frames=track_timeout_frames,
        stop_speed_mps=stop_speed_mps,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )


def main():
    ap = argparse.ArgumentParser(description="Traffic patterns with tracking, ROI paths, and speeds")
    ap.add_argument("input_video")
    ap.add_argument("--model", default="models/yolov8n.pt")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--run", default="patterns_run1")
    ap.add_argument("--tracker", default=DEFAULT_TRACKER,
                    help="bytetrack.yaml, botsort.yaml, or a path to a tracker YAML")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--rois", required=True, help="rois.json (may include 'calibration')")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--device", default=None, help="'cpu', 'auto', or a CUDA index such as 0")
    ap.add_argument("--no-video", action="store_true", help="Skip the annotated video (much faster)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run_analysis(
        input_video=args.input_video, model=args.model,
        output=args.output, run=args.run, rois_path=args.rois,
        tracker=args.tracker, imgsz=args.imgsz,
        conf=args.conf, iou=args.iou, device=args.device,
        write_video=not args.no_video,
        progress_callback=lambda f, m: print(f"\r{m}", end="", flush=True),
    )
    print()
    print(f"Per-track metrics: {out['tracks_csv']}")
    print(f"Movement (O-D) counts: {out['movements_csv']}")
    if out["video"]:
        print(f"Annotated video: {out['video']}")


if __name__ == "__main__":
    main()
