# studio/qaqc.py
from __future__ import annotations
from pathlib import Path
import itertools, re, csv, hashlib, shutil
from typing import List
from .logutil import setup_logger
log = setup_logger()

def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        return [ln.rstrip("\n\r") for ln in f.readlines()]

def _write_lines(path: Path, lines: list[str], dry_run: bool) -> bool:
    if dry_run:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='\n', encoding='utf-8') as f:
        for o in lines:
            f.write(o + '\n')
    return True

def _parse_yolo_line(s: str):
    parts = s.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(parts[0])
        nums = [float(x) for x in parts[1:5]]
        return cls, nums
    except ValueError:
        return None

def _near_equal(a: list[float], b: list[float], eps: float) -> bool:
    return len(a) == len(b) and all(abs(x-y) <= eps for x,y in zip(a,b))

def validate_labels(labels_dir: Path) -> dict:
    labels_dir = Path(labels_dir)
    issues_csv = labels_dir / "issues.csv"
    total_files = 0
    issues = {
        "malformed": 0,
        "non_numeric": 0,
        "out_of_bounds": 0,
        "non_positive_wh": 0,
        "empty_files": 0,
    }
    rows = [["file","line_no","issue","text"]]
    for path in sorted(labels_dir.rglob("*.txt")):
        total_files += 1
        lines = _read_lines(path)
        if not lines:
            issues["empty_files"] += 1
            rows.append([str(path), "", "empty_file", ""])
            continue
        for i, ln in enumerate(lines, 1):
            parts = ln.strip().split()
            if len(parts) < 5:
                issues["malformed"] += 1
                rows.append([str(path), i, "malformed", ln]); continue
            try:
                cls = int(parts[0]); x,y,w,h = map(float, parts[1:5])
            except (ValueError, IndexError):
                issues["non_numeric"] += 1
                rows.append([str(path), i, "non_numeric", ln]); continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                issues["out_of_bounds"] += 1
                rows.append([str(path), i, "out_of_bounds", ln])
            if w <= 0 or h <= 0:
                issues["non_positive_wh"] += 1
                rows.append([str(path), i, "non_positive_wh", ln])
    with issues_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)
    log.info("Validation finished: %s", issues)
    log.info("Issue details -> %s", issues_csv)
    return {"total_files": total_files, **issues, "issues_csv": str(issues_csv)}

def cleanup_labels(labels_dir: Path, *, delete_lines: list[str]|None=None,
                   delete_classes: list[int]|None=None,
                   delete_regex: list[str]|None=None,
                   delete_near: list[str]|None=None,
                   eps: float=0.0, dry_run: bool=False) -> dict:
    delete_lines = [s.strip() for s in (delete_lines or []) if s.strip()]
    delete_classes = set(delete_classes or [])
    delete_regex = [re.compile(p) for p in (delete_regex or [])]
    near_specs = []
    for s in (delete_near or []):
        parsed = _parse_yolo_line(s)
        if parsed is not None:
            near_specs.append(parsed)
    def should_delete(line: str) -> bool:
        raw = line.rstrip("\n\r")
        stripped = raw.strip()
        if stripped in delete_lines:
            return True
        for rgx in delete_regex:
            if rgx.search(raw):
                return True
        parts = stripped.split()
        if parts:
            try:
                cls = int(parts[0])
                if cls in delete_classes:
                    return True
            except ValueError:
                pass
        parsed = _parse_yolo_line(stripped)
        if parsed is not None:
            cls, nums = parsed
            for tgt_cls, tgt_nums in near_specs:
                if cls == tgt_cls and _near_equal(nums, tgt_nums, eps):
                    return True
        return False
    total_files = 0; total_removed = 0
    for fpath in sorted(Path(labels_dir).rglob("*.txt")):
        existing = _read_lines(fpath)
        if not existing:
            continue
        kept = [ln for ln in existing if not should_delete(ln)]
        removed = len(existing) - len(kept)
        if removed > 0:
            total_files += 1; total_removed += removed
            _write_lines(fpath, kept, dry_run)
            log.info("delete: %3d removed -> %s", removed, fpath)
    if dry_run:
        log.info("Dry run: no files modified.")
    return {"files_modified": total_files, "lines_removed": total_removed, "dry_run": dry_run}

def dedupe_labels(labels_dir: Path, dry_run: bool=False) -> dict:
    files = 0; removed = 0
    for fpath in sorted(Path(labels_dir).rglob("*.txt")):
        lines = _read_lines(fpath)
        if not lines:
            continue
        seen = set(); out = []
        for ln in lines:
            if ln in seen:
                removed += 1; continue
            seen.add(ln); out.append(ln)
        if len(out) != len(lines):
            files += 1
            _write_lines(fpath, out, dry_run)
            log.info("dedupe: %3d duplicates removed -> %s", len(lines)-len(out), fpath)
    if dry_run:
        log.info("Dry run: no files modified.")
    return {"files_modified": files, "lines_removed": removed, "dry_run": dry_run}

def _stable_hash(s: str) -> int:
    import hashlib
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)

def make_site_disjoint_splits(images_root: Path, labels_root: Path, out_root: Path, *,
                              train=0.7, val=0.2, test=0.1,
                              site_regex: str|None=None,
                              copy: bool=True,
                              classes_txt: Path|None=None) -> dict:
    images_root = Path(images_root); labels_root = Path(labels_root); out_root = Path(out_root)
    img_paths = [p for p in images_root.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]
    if not img_paths:
        raise RuntimeError(f"No images found under {images_root}")
    site_pat = re.compile(site_regex) if site_regex else None
    pairs = []
    for ip in img_paths:
        if site_pat:
            m = site_pat.search(str(ip))
            site = m.group(1) if m else ip.parent.name
        else:
            site = ip.parent.name
        lp = labels_root / ip.with_suffix(".txt").name
        if not lp.exists():
            lp = None
        pairs.append((site, ip, lp))
    by_site = {}
    for site, ip, lp in pairs:
        by_site.setdefault(site, []).append((ip, lp))
    sites = sorted(by_site.keys())
    sites_sorted = sorted(sites, key=lambda s: _stable_hash(s))
    n = len(sites_sorted)
    n_train = round(n * train); n_val = round(n * val); n_test = n - n_train - n_val
    site_splits = {
        "train": set(sites_sorted[:n_train]),
        "val": set(sites_sorted[n_train:n_train+n_val]),
        "test": set(sites_sorted[n_train+n_val:]),
    }
    import shutil
    for split, site_set in site_splits.items():
        for ip, lp in itertools.chain.from_iterable(by_site[s] for s in site_set):
            dst_img = out_root / "images" / split / ip.name
            dst_lbl = out_root / "labels" / split / (ip.with_suffix(".txt").name)
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            if copy:
                shutil.copy2(ip, dst_img)
                if lp and lp.exists():
                    shutil.copy2(lp, dst_lbl)
                else:
                    dst_lbl.touch()
            else:
                try:
                    dst_img.symlink_to(ip)
                except FileExistsError:
                    pass
                if lp and lp.exists():
                    try:
                        dst_lbl.symlink_to(lp)
                    except FileExistsError:
                        pass
                else:
                    dst_lbl.touch()
    names = []
    if classes_txt and Path(classes_txt).exists():
        names = [ln.strip() for ln in Path(classes_txt).read_text().splitlines() if ln.strip()]
    data_yaml = out_root / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {out_root.resolve()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        if names:
            f.write("names:\n")
            for i, n in enumerate(names):
                f.write(f"  {i}: {n}\n")
    log.info("Created site-disjoint dataset at %s (sites=%d)", out_root, len(sites))
    return {"out_root": str(out_root), "sites": len(sites), "data_yaml": str(data_yaml)}

def write_fhwa_mapping_csv(out_csv: Path, *, include_rollup=True):
    rows = [
        ["label","fhwa_class","hpms_group"],
        ["pedestrian","",""],
        ["cyclist","",""],
        ["motorcycle","1","Motorcycles"],
        ["passenger_car","2","Passenger Cars"],
        ["other_2ax_4tire","3","Passenger Cars"],
        ["bus","4","Buses"],
        ["2ax_6tire_single","5","Single-Unit Trucks"],
        ["3ax_single","6","Single-Unit Trucks"],
        ["4+ax_single","7","Single-Unit Trucks"],
        ["<=4ax_single_trailer","8","Combination Trucks"],
        ["5ax_single_trailer","9","Combination Trucks"],
        ["6+ax_single_trailer","10","Combination Trucks"],
        ["<=5ax_multi_trailer","11","Combination Trucks"],
        ["6ax_multi_trailer","12","Combination Trucks"],
        ["7+ax_multi_trailer","13","Combination Trucks"],
    ]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerows(rows)
    log.info("Wrote FHWA mapping template -> %s", out_csv)
    return str(out_csv)
