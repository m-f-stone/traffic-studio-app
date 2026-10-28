"""Tests for the torch-free parts of the tracking script: ROI loading, speed and
movement features, track finalization, and the annotated-video writer."""
import json
import shutil

import numpy as np
import pytest

from third_party import traffic_analysis_tracking as t


def test_default_tracker_is_one_ultralytics_ships():
    assert t.DEFAULT_TRACKER in t.SUPPORTED_TRACKERS
    assert "strongsort" not in t.DEFAULT_TRACKER


def test_compress_seq_and_movement_label_are_ascii():
    assert t.compress_seq(["a", "a", "b", "b", "a"]) == ["a", "b", "a"]
    assert t.movement_label([]) == "unknown"
    assert t.movement_label(["a"]) == "within a"
    assert t.movement_label(["a", "b", "a"]) == "u/loop in a"
    label = t.movement_label(["north_in", "east_out"])
    assert label == "north_in -> east_out"
    label.encode("cp1252")   # must survive a Windows default-encoding CSV write


def test_compute_speeds_counts_stops_only_when_calibrated():
    ts = [0, 1, 2, 3, 4, 5, 6]
    xs = [0, 1, 1, 1, 1, 2, 3]   # one 1-unit step, a 4-sample standstill, then moving
    ys = [0] * 7
    speeds, avg, mx, stops = t.compute_speeds(xs, ys, ts, stop_speed=0.2)
    assert len(speeds) == 6 and mx == 1.0 and stops == 1
    _, _, _, stops_px = t.compute_speeds(xs, ys, ts, stop_speed=None)
    assert stops_px == 0


def test_finalize_track_row_units_and_path():
    st = t.TrackState(2, "car", frame_idx=10, conf=0.5)
    for i, (x, y, roi) in enumerate([(0, 0, "north_in"), (3, 4, "north_in"), (6, 8, "east_out")]):
        st.times.append(i / 30.0)
        st.px.append(x)
        st.py.append(y)
        if not st.rois or st.rois[-1] != roi:
            st.rois.append(roi)
    st.last_frame = 12
    row = t.finalize_track(7, st, fps=30.0, world_mode=False)
    assert row[:4] == [7, "car", 10, 12]
    assert row[5:8] == ["north_in", "east_out", "north_in -> east_out"]
    assert row[8:10] == [10.0, "px"] and row[12] == "px/s" and row[13] == 0
    assert t.finalize_track(1, t.TrackState(0, "x", 0, 0.1), 30.0, False) is None


def test_load_rois_skips_degenerate_polygons_and_strips_calibration(tmp_path):
    p = tmp_path / "rois.json"
    p.write_text(json.dumps({
        "north": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        "line": [[0, 0], [5, 5]],
        "calibration": {"meters_per_pixel": 0.05},
    }))
    rois, calib = t.load_rois(p)
    assert set(rois) == {"north"} and rois["north"].area == 100
    assert calib == {"meters_per_pixel": 0.05}
    H, use_H, mpp = t.build_homography(calib)
    assert (H, use_H, mpp) == (None, False, 0.05)
    assert t.px_to_world((10, 20), None, 0.05) == (0.5, 1.0, True)


def test_resolve_device_without_torch_is_cpu():
    assert t.resolve_device(None) == "cpu"
    assert t.resolve_device("auto") == "cpu"
    assert t.resolve_device("cpu") == "cpu"
    assert t.resolve_device("0") == "cpu"   # no CUDA here: falls back instead of raising


def _frames(n, w=64, h=48):
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, : (i * 7) % w] = (0, 255, 0)
        yield f


def test_video_writer_opencv_fallback(tmp_path):
    out = tmp_path / "cv.mp4"
    w = t.AnnotatedVideoWriter(out, 64, 48, 10.0, prefer_ffmpeg=False)
    assert w.backend == "opencv/mp4v"
    for f in _frames(5):
        w.write(f)
    w.release()
    assert w.frames == 5 and out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_video_writer_ffmpeg_pipe_handles_odd_dimensions(tmp_path):
    out = tmp_path / "ff.mp4"
    w = t.AnnotatedVideoWriter(out, 63, 47, 10.0)
    assert w.backend == "ffmpeg/libx264"
    for f in _frames(8, 63, 47):
        w.write(f)
    w.release()
    assert w.frames == 8 and out.exists() and out.stat().st_size > 0
    import cv2
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    cap.release()
