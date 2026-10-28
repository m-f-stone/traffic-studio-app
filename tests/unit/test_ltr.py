"""Tests for approach detection and left/through/right post-processing."""
import csv

from studio.analyzer import _detect_approach, _ltr_from, _postprocess_ltr


def test_detect_approach_variants():
    assert _detect_approach("NB Lane 1") == "N"
    assert _detect_approach("south_in") == "S"
    assert _detect_approach("E Roadway") == "E"
    assert _detect_approach("westlane") == "W"
    assert _detect_approach("Sidewalk") is None


def test_ltr_from_pairs():
    assert _ltr_from("north_in", "east_out") == ("N", "Right")
    assert _ltr_from("north_in", "west_out") == ("N", "Left")
    assert _ltr_from("north_in", "south_out") == ("N", "Through")
    assert _ltr_from("north_in", "north_out") == ("N", "U/Loop")
    assert _ltr_from("sidewalk", "east_out") == (None, None)


def test_postprocess_ltr_writes_per_track_and_aggregate(tmp_path):
    tracks = tmp_path / "run_tracks.csv"
    with tracks.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "class", "entry_roi", "exit_roi", "roi_path"])
        w.writerow([1, "car", "north_in", "east_out", "north_in -> east_out"])
        w.writerow([2, "car", "north_in", "east_out", "north_in -> east_out"])
        w.writerow([3, "truck", "south_in", "north_out", "south_in -> north_out"])
        w.writerow([4, "car", "", "", ""])
    _postprocess_ltr(tmp_path, "run")

    rows = list(csv.DictReader((tmp_path / "run_tracks_ltr.csv").open(encoding="utf-8")))
    assert [r["movement_ltr"] for r in rows] == ["Right", "Right", "Through", "Unknown"]
    assert rows[0]["approach"] == "N"

    agg = list(csv.reader((tmp_path / "run_movements_ltr.csv").open(encoding="utf-8")))
    assert agg[0] == ["approach", "turn", "class", "count"]
    assert ["N", "Right", "car", "2"] in agg
    assert ["S", "Through", "truck", "1"] in agg
