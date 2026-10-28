#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


# -------------------------
# ROI loading & calibration
# -------------------------
def load_rois(roi_path):
    """
    ROI JSON format:
    {
      "NS Roadway": [[x,y], ...],
      "EW Roadway": [[x,y], ...],
      "East Sidewalk": [[x,y], ...],
      ...
      // optional calibration block (either-or; both optional)
      "calibration": {
         "meters_per_pixel": 0.024   // simple scale
         // OR
         "image_points": [[x,y], ... 4+ pts],
         "world_points_m": [[X,Y], ... matching 4+ pts in meters]
      }
    }
    """
    with open(roi_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    calib = raw.pop("calibration", None)

    # Clean duplicate end-points if any
    def clean_pts(pts):
        uniq = []
        for x, y in pts:
            if not uniq or (uniq[-1][0] != x or uniq[-1][1] != y):
                uniq.append((x, y))
        if len(uniq) > 1 and uniq[0] == uniq[-1]:
            uniq.pop()
        return uniq

    rois = {name: Polygon(clean_pts(pts)) for name, pts in raw.items()}
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
    """
    Convert pixel (x,y) to world (X,Y) in meters, if calibration is available.
    If none, return pixel coords and flag that it's pixel space.
    """
    x, y = pt_xy
    if H is not None:
        p = np.array([ [x, y, 1.0] ], dtype=np.float32).T
        pw = H @ p
        pw = pw / (pw[2] + 1e-9)
        return float(pw[0]), float(pw[1]), True
    elif meters_per_px is not None:
        return x * meters_per_px, y * meters_per_px, True
    else:
        return float(x), float(y), False


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
        self.times = []  # list of t_sec for each sample
        self.px = []     # x in world or pixel
        self.py = []     # y in world or pixel
        self.rois = []   # ROI name sequence (dedup)
        self.max_conf = float(conf)
        self.trail = deque(maxlen=30)  # for on-video breadcrumbs


def compress_seq(seq):
    out = []
    prev = None
    for s in seq:
        if s != prev:
            out.append(s)
            prev = s
    return out


def movement_label(path_seq):
    """Return a simple movement label from ROI path sequence."""
    if not path_seq:
        return "unknown"
    if len(path_seq) == 1:
        return f"within {path_seq[0]}"
    if path_seq[0] == path_seq[-1]:
        return f"u/loop in {path_seq[0]}"
    return f"{path_seq[0]} → {path_seq[-1]}"


def compute_speeds(xs, ys, ts):
    """Return per-sample speed (m/s or px/s), avg and max; stop segments."""
    if len(ts) < 2:
        return [], 0.0, 0.0, 0

    speeds = []
    stops = 0
    stop_run = 0
    for i in range(1, len(ts)):
        dt = max(ts[i] - ts[i-1], 1e-9)
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        v = math.hypot(dx, dy) / dt
        speeds.append(v)
        if v < 0.2:  # <0.2 m/s considered stopped (adjust if using px/s)
            stop_run += 1
        else:
            if stop_run >= 3:  # e.g., ≥3 consecutive samples
                stops += 1
            stop_run = 0
    if stop_run >= 3:
        stops += 1

    avg_v = float(np.mean(speeds)) if speeds else 0.0
    max_v = float(np.max(speeds)) if speeds else 0.0
    return speeds, avg_v, max_v, stops


def streaming_patterns(
    model_path, input_video, output_dir, run_name,
    tracker_cfg="strongsort.yaml", imgsz=640,
    roi_polygons=None, calib=None, conf=0.3, iou=0.5, device=None
):
    """
    Stream detection+tracking, build trajectories & behavior features,
    save per-track CSV + annotated video.
    """
    model = YOLO(model_path)
    if device is not None:
        model.to(device)

    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv = out_dir / f"{run_name}_tracks.csv"
    moves_csv  = out_dir / f"{run_name}_movements.csv"
    video_out  = out_dir / f"{run_name}_annotated.mp4"
    CSV_ENCODING = "utf-8-sig" 

    # Calibration
    H, use_H, mpp = build_homography(calib or {})
    world_mode = use_H or (mpp is not None)

    # Video info (fps, size)
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer = None
    class_names = model.names

    # Active tracks: tid -> TrackState
    active = {}
    # Once a track disappears for > buffer frames, we finalize it.
    last_seen = {}  # tid -> frame_idx
    end_buffer = 30  # frames

    def roi_name_pixel(cx, cy):
        if not roi_polygons:
            return None
        p = Point(float(cx), float(cy))
        for rn, poly in roi_polygons.items():
            if poly.contains(p):
                return rn
        return None

    frame_time = 0.0

    # CSV headers
    with tracks_csv.open("w", newline="", encoding=CSV_ENCODING) as ftrk, moves_csv.open("w", newline="", encoding=CSV_ENCODING) as fmov:
        tw = csv.writer(ftrk); mw = csv.writer(fmov)
        tw.writerow([
            "track_id","class","first_frame","last_frame","duration_s",
            "entry_roi","exit_roi","roi_path",
            "distance","distance_units","avg_speed","max_speed","speed_units",
            "stops"
        ])
        mw.writerow(["movement","class","count"])

        for frame_idx, result in enumerate(
            model.track(
                source=input_video, stream=True, imgsz=imgsz,
                tracker=tracker_cfg, conf=conf, iou=iou, persist=True
            ), start=0
        ):
            if writer is None:
                h, w = result.orig_shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_out), fourcc, fps, (w, h))

            # Draw base frame
            frame_rgb = result.plot()  # RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Update tracks
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    if box.id is None:  # not assigned a track id yet
                        continue
                    tid = int(box.id[0])
                    cls = int(box.cls[0])
                    confd = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

                    # world or pixel coords for speed
                    X, Y, _ = px_to_world((cx, cy), H, mpp)

                    if tid not in active:
                        active[tid] = TrackState(cls, class_names.get(cls, str(cls)), frame_idx, confd)
                    st = active[tid]
                    st.last_frame = frame_idx
                    st.max_conf = max(st.max_conf, confd)
                    st.times.append(frame_time)
                    st.px.append(X); st.py.append(Y)
                    st.trail.append((int(cx), int(cy)))  # draw in pixels

                    # ROI sequence using pixel coords
                    rn = roi_name_pixel(cx, cy)
                    if rn is not None:
                        if not st.rois or st.rois[-1] != rn:
                            st.rois.append(rn)

                    last_seen[tid] = frame_idx

            # Annotate trails + labels
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    if box.id is None: 
                        continue
                    tid = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"ID {tid}"
                    if tid in active:
                        path_seq = compress_seq(active[tid].rois)
                        move = movement_label(path_seq)
                        label = f"{label} | {move}"
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame_bgr, label, (x1, max(20, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Draw trails
            for tid, st in list(active.items()):
                pts = list(st.trail)
                if len(pts) >= 2:
                    cv2.polylines(frame_bgr, [np.array(pts, dtype=np.int32)], False, (0,200,255), 2)

            writer.write(frame_bgr)
            frame_time += 1.0 / fps

            # Finalize tracks that vanished
            ended = []
            for tid, lastf in list(last_seen.items()):
                if frame_idx - lastf > end_buffer:
                    ended.append(tid)

            for tid in ended:
                st = active.pop(tid, None)
                last_seen.pop(tid, None)
                if st is None or len(st.times) < 2:
                    continue

                # Features
                speeds, avg_v, max_v, stops = compute_speeds(st.px, st.py, st.times)
                # distance (sum of segment lengths)
                dist = 0.0
                for i in range(1, len(st.px)):
                    dist += math.hypot(st.px[i]-st.px[i-1], st.py[i]-st.py[i-1])
                duration = (st.last_frame - st.first_frame + 1) / fps

                path_seq = compress_seq(st.rois)
                entry = path_seq[0] if path_seq else ""
                exitr = path_seq[-1] if path_seq else ""
                move = movement_label(path_seq)

                unit_dist = "m" if world_mode else "px"
                unit_speed = "m/s" if world_mode else "px/s"

                # Write track row
                tw.writerow([
                    tid, st.name, st.first_frame, st.last_frame, round(duration,3),
                    entry, exitr, " -> ".join(path_seq),
                    round(dist,3), unit_dist,
                    round(avg_v,3), round(max_v,3), unit_speed,
                    stops
                ])

        # flush any remaining active tracks at end-of-video
        for tid, st in active.items():
            if len(st.times) < 2:
                continue
            speeds, avg_v, max_v, stops = compute_speeds(st.px, st.py, st.times)
            dist = 0.0
            for i in range(1, len(st.px)):
                dist += math.hypot(st.px[i]-st.px[i-1], st.py[i]-st.py[i-1])
            duration = (st.last_frame - st.first_frame + 1) / fps
            path_seq = compress_seq(st.rois)
            entry = path_seq[0] if path_seq else ""
            exitr = path_seq[-1] if path_seq else ""
            unit_dist = "m" if world_mode else "px"
            unit_speed = "m/s" if world_mode else "px/s"
            tw.writerow([
                tid, st.name, st.first_frame, st.last_frame, round(duration,3),
                entry, exitr, " -> ".join(path_seq),
                round(dist,3), unit_dist,
                round(np.mean(speeds) if speeds else 0.0,3),
                round(np.max(speeds) if speeds else 0.0,3),
                unit_speed, stops
            ])

    if writer is not None:
        writer.release()

    # Aggregate movements (simple O-D from ROI path)
    counts = defaultdict(lambda: defaultdict(int))  # movement -> class -> count
    with tracks_csv.open("r", newline="", encoding=CSV_ENCODING) as f:
        r = csv.DictReader(f)
        for row in r:
            move = movement_label(row["roi_path"].split(" -> ") if row["roi_path"] else [])
            counts[move][row["class"]] += 1
    with moves_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["movement","class","count"])
        for move, per_cls in counts.items():
            for cls, n in per_cls.items():
                w.writerow([move, cls, n])

    return tracks_csv, moves_csv, video_out


def run_analysis(input_video, model, output, run, rois_path,
                 tracker="strongsort.yaml", imgsz=640,
                 conf=0.30, iou=0.50, device=None):
    """Direct-call API for running traffic analysis (no sys.argv manipulation).

    Returns:
        tuple of (tracks_csv, moves_csv, video_out) Paths
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
        device=device
    )


def main():
    ap = argparse.ArgumentParser(description="Traffic patterns with tracking, ROI paths, and speeds")
    ap.add_argument("input_video")
    ap.add_argument("--model", default="models/yolov8n.pt")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--run", default="patterns_run1")
    ap.add_argument("--tracker", default="strongsort.yaml", help="strongsort.yaml or bytetrack.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--rois", required=True, help="rois.json (may include 'calibration')")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou",  type=float, default=0.50)
    ap.add_argument("--device", default=None, help="e.g., '0' for CUDA:0")
    args = ap.parse_args()

    tracks_csv, moves_csv, video_out = run_analysis(
        input_video=args.input_video, model=args.model,
        output=args.output, run=args.run, rois_path=args.rois,
        tracker=args.tracker, imgsz=args.imgsz,
        conf=args.conf, iou=args.iou, device=args.device
    )
    print(f"Per-track metrics: {tracks_csv}")
    print(f"Movement (O-D) counts: {moves_csv}")
    print(f"Annotated video: {video_out}")


if __name__ == "__main__":
    main()
