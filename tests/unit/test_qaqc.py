"""Tests for the label QA/QC helpers (pure Python, no PyTorch needed)."""
from pathlib import Path

from studio import qaqc


def _make_dataset(root: Path, sites=("A", "B", "C"), per_site=3):
    img = root / "images"
    lbl = root / "labels"
    lbl.mkdir(parents=True, exist_ok=True)
    for s in sites:
        (img / f"site_{s}").mkdir(parents=True, exist_ok=True)
        for i in range(per_site):
            (img / f"site_{s}" / f"img_{s}_{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (lbl / f"img_{s}_{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (root / "classes.txt").write_text("car\ntruck\n")
    return img, lbl


def test_data_yaml_is_multiline_and_parseable(tmp_path):
    img, lbl = _make_dataset(tmp_path)
    res = qaqc.make_site_disjoint_splits(img, lbl, tmp_path / "out", classes_txt=tmp_path / "classes.txt")

    text = Path(res["data_yaml"]).read_text()
    assert "\\n" not in text, "literal backslash-n sequences in data.yaml"
    lines = text.splitlines()
    assert lines[0].startswith("path: ")
    assert "train: images/train" in lines
    assert "val: images/val" in lines
    assert "test: images/test" in lines
    assert "names:" in lines
    assert "  0: car" in lines and "  1: truck" in lines


def test_splits_are_site_disjoint(tmp_path):
    img, lbl = _make_dataset(tmp_path, sites=("A", "B", "C", "D", "E"))
    out = tmp_path / "out"
    qaqc.make_site_disjoint_splits(img, lbl, out, train=0.6, val=0.2, test=0.2)

    sites_by_split = {}
    for split in ("train", "val", "test"):
        names = [p.stem for p in (out / "images" / split).glob("*.jpg")]
        sites_by_split[split] = {n.split("_")[1] for n in names}
        # every image has a label file alongside it
        for n in names:
            assert (out / "labels" / split / f"{n}.txt").exists()
    assert sites_by_split["train"].isdisjoint(sites_by_split["val"])
    assert sites_by_split["train"].isdisjoint(sites_by_split["test"])
    assert sites_by_split["val"].isdisjoint(sites_by_split["test"])


def test_cleanup_labels_by_class_regex_and_near(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n2 0.9 0.9 0.05 0.05\n3 0.1 0.1 0.3 0.3\n")
    res = qaqc.cleanup_labels(tmp_path, delete_classes=[1], delete_regex=[r"^2 "],
                              delete_near=["3 0.1 0.1 0.3 0.3"], eps=1e-6, dry_run=False)
    assert res == {"files_modified": 1, "lines_removed": 3, "dry_run": False}
    assert f.read_text().splitlines() == ["0 0.5 0.5 0.1 0.1"]


def test_cleanup_dry_run_leaves_files_alone(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    res = qaqc.cleanup_labels(tmp_path, delete_classes=[1], dry_run=True)
    assert res["lines_removed"] == 1 and res["dry_run"] is True
    assert len(f.read_text().splitlines()) == 2


def test_dedupe_labels(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    res = qaqc.dedupe_labels(tmp_path)
    assert res["lines_removed"] == 1
    assert len(f.read_text().splitlines()) == 2


def test_validate_labels_reports_issue_types(tmp_path):
    (tmp_path / "ok.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (tmp_path / "bad.txt").write_text("0 0.5\n0 x 0.5 0.1 0.1\n0 1.5 0.5 0.1 0.1\n0 0.5 0.5 0 0.1\n")
    (tmp_path / "empty.txt").write_text("")
    res = qaqc.validate_labels(tmp_path)
    assert res["total_files"] == 3
    assert res["malformed"] == 1
    assert res["non_numeric"] == 1
    assert res["out_of_bounds"] == 1
    assert res["non_positive_wh"] == 1
    assert res["empty_files"] == 1
    assert Path(res["issues_csv"]).exists()
