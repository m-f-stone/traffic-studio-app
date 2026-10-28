"""Tests for ROI loading, masking (parallel), and border-crop relabelling."""
import json
import threading

import cv2
import numpy as np
import pytest

from studio import roi_utils


# ---------------------------------------------------------------- relabel math
def test_relabel_centered_box_shifts_and_renormalizes():
    # 100x100 image, 10% margin -> crop 80x80 at (10,10)
    lines = roi_utils.relabel_for_crop("0 0.5 0.5 0.2 0.2\n", 100, 100, 10, 10, 80, 80)
    assert lines == ["0 0.500000 0.500000 0.250000 0.250000"]


def test_relabel_drops_boxes_centered_outside_crop_and_clips_edges():
    text = "0 0.02 0.5 0.1 0.1\n1 0.15 0.5 0.2 0.2\nbad line\n"
    lines = roi_utils.relabel_for_crop(text, 100, 100, 10, 10, 80, 80)
    assert len(lines) == 1
    cls, x, y, w, h = lines[0].split()
    assert cls == "1"
    # box 1: centre (15,50) px -> (5,40) in crop; half-width 10 -> clipped at x=0 -> width 15 px
    assert abs(float(w) - 15 / 80) < 1e-6
    assert abs(float(x) - (7.5 / 80)) < 1e-6


def test_relabel_drops_tiny_boxes():
    assert roi_utils.relabel_for_crop("0 0.5 0.5 0.001 0.001\n", 100, 100, 10, 10, 80, 80) == []


# ---------------------------------------------------------------- ROI JSON
def test_load_roi_any_handles_both_schemas(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"north": [[0, 0], [10, 0], [10, 10]],
                                  "calibration": {"meters_per_pixel": 0.1}}))
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"meta": {"w": 10}, "rois": {
        "north": {"points": [[0, 0], [10, 0], [10, 10]], "include": True},
        "lane_1": {"points": [[0, 0], [5, 0], [5, 5]], "include": False}}}))

    assert roi_utils.load_roi_any(legacy) == {"north": {"points": [[0, 0], [10, 0], [10, 10]], "include": True}}
    rois = roi_utils.load_roi_any(new)
    assert rois["lane_1"]["include"] is False
    assert set(roi_utils.included_polygons(rois)) == {"north"}


def test_normalize_for_tracking_can_force_lane_rois(tmp_path):
    src = tmp_path / "t.json"
    src.write_text(json.dumps({"rois": {
        "north": {"points": [[0, 0], [10, 0], [10, 10]], "include": True},
        "north_lane_1": {"points": [[0, 0], [5, 0], [5, 5]], "include": False}}}))
    out = roi_utils.normalize_for_tracking(src, tmp_path / "flat.json")
    assert set(json.loads(out.read_text())) == {"north"}
    out = roi_utils.normalize_for_tracking(src, tmp_path / "flat2.json", include_lanes=True)
    assert set(json.loads(out.read_text())) == {"north", "north_lane_1"}


# ---------------------------------------------------------------- masks
def _write_images(d, n, size=(40, 30)):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((size[1], size[0], 3), 200, np.uint8)
        cv2.imwrite(str(d / f"f_{i:03d}.png"), img)


def test_mask_from_rois_fills_polygon_union():
    mask = roi_utils.mask_from_rois({"a": [[0, 0], [10, 0], [10, 10], [0, 10]],
                                     "b": [[20, 20], [30, 20], [30, 30], [20, 30]],
                                     "degenerate": [[1, 1], [2, 2]]}, 40, 40)
    assert mask.shape == (40, 40)
    assert mask[5, 5] == 255 and mask[25, 25] == 255 and mask[15, 15] == 0


def test_mask_images_parallel_matches_serial_and_scales_mismatched_mask(tmp_path):
    src = tmp_path / "src"
    _write_images(src, 12)
    roi = tmp_path / "roi.json"
    roi.write_text(json.dumps({"keep": [[0, 0], [20, 0], [20, 30], [0, 30]]}))
    mask = roi_utils.mask_for_images(src, roi)
    assert mask.shape == (30, 40)

    res = roi_utils.mask_images(src, tmp_path / "out", mask, workers=4)
    assert res == {"images": 12, "skipped": 0, "cancelled": False}
    out = cv2.imread(str(tmp_path / "out" / "f_000.png"))
    assert out[15, 5].tolist() == [200, 200, 200]   # inside ROI kept
    assert out[15, 35].tolist() == [0, 0, 0]         # outside ROI blacked out

    # a mask built for another frame size is scaled, not rejected
    small_mask = cv2.resize(mask, (20, 15), interpolation=cv2.INTER_NEAREST)
    assert roi_utils.apply_mask_to_image(src / "f_001.png", tmp_path / "scaled.png", small_mask)
    scaled = cv2.imread(str(tmp_path / "scaled.png"))
    assert scaled.shape == (30, 40, 3)
    assert scaled[15, 5].tolist() == [200, 200, 200] and scaled[15, 35].tolist() == [0, 0, 0]


def test_mask_for_images_errors_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        roi_utils.mask_for_images(tmp_path / "nope", tmp_path / "roi.json")
    src = tmp_path / "src"
    _write_images(src, 1)
    roi = tmp_path / "roi.json"
    roi.write_text(json.dumps({"rois": {"x": {"points": [[0, 0], [1, 0], [1, 1]], "include": False}}}))
    with pytest.raises(ValueError):
        roi_utils.mask_for_images(src, roi)


def test_run_parallel_reports_progress_and_cancels(tmp_path):
    seen = []
    results, cancelled = roi_utils.run_parallel(range(50), lambda i: i * 2, workers=2,
                                                progress_callback=lambda f, m: seen.append(f))
    assert sorted(results) == [i * 2 for i in range(50)] and not cancelled
    assert seen and seen[-1] == 1.0

    gate = threading.Event()
    results, cancelled = roi_utils.run_parallel(range(50), lambda i: gate.wait(0.01) or i, workers=2,
                                                cancel_check=lambda: True)
    assert cancelled and len(results) < 50


def test_crop_margin_and_relabel_parallel(tmp_path):
    src = tmp_path / "images"
    _write_images(src, 6, size=(100, 100))
    labels = tmp_path / "labels"
    labels.mkdir()
    for i in range(6):
        (labels / f"f_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.02 0.5 0.1 0.1\n")
    n_img, n_lbl = roi_utils.crop_margin_and_relabel(src, tmp_path / "cropped", labels, workers=3, margin_frac=0.10)
    assert (n_img, n_lbl) == (6, 6)
    assert cv2.imread(str(tmp_path / "cropped" / "f_000.png")).shape == (80, 80, 3)
    out_lbl = tmp_path / "labels" / "cropped" / "f_000.txt"
    assert out_lbl.read_text().splitlines() == ["0 0.500000 0.500000 0.250000 0.250000"]
