#!/usr/bin/env python3
import argparse, os

def main():
    ap = argparse.ArgumentParser(
        description="Append (or overwrite) YOLO labels to a range of frames."
    )
    ap.add_argument("--labels_dir", required=True,
                    help="Folder that holds .txt label files (e.g. data/labels/train)")
    ap.add_argument("--start", type=int, required=True, help="First frame index (inclusive)")
    ap.add_argument("--end",   type=int, required=True, help="Last frame index (inclusive)")
    ap.add_argument("--step",  type=int, default=1, help="Step between frames (default 1)")
    ap.add_argument("--prefix", default="frame_", help="Filename prefix (default: frame_)")
    ap.add_argument("--pad", type=int, default=5, help="Zero-padding width (default: 5)")
    ap.add_argument("--ext", default=".txt", help="Label file extension (default: .txt)")
    ap.add_argument("--line", action="append", required=True,
                    help=("YOLO label line to add, e.g. "
                          "'0 0.293750 0.181481 0.014583 0.044444'. "
                          "Use --line multiple times to add several."))
    ap.add_argument("--mode", choices=["append","overwrite"], default="append",
                    help="append (default) or overwrite label files")
    ap.add_argument("--dedupe", action="store_true",
                    help="Avoid adding a line if it's already present in the file")
    args = ap.parse_args()

    os.makedirs(args.labels_dir, exist_ok=True)
    lines_to_add = [l.strip() for l in args.line if l.strip()]
    if not lines_to_add:
        raise SystemExit("No non-empty --line values provided.")

    total = 0
    for i in range(args.start, args.end + 1, args.step):
        fname = f"{args.prefix}{i:0{args.pad}}{args.ext}"
        fpath = os.path.join(args.labels_dir, fname)

        if args.mode == "overwrite" or not os.path.exists(fpath):
            existing = []
        else:
            with open(fpath, "r", newline="") as f:
                existing = [ln.rstrip("\n\r") for ln in f.readlines()]

        out = list(existing)
        for ln in lines_to_add:
            if args.mode == "overwrite":
                # in overwrite mode we rebuild file only from provided lines
                pass
            else:
                if args.dedupe and ln in existing:
                    continue
                # otherwise append
                out.append(ln)

        if args.mode == "overwrite":
            out = lines_to_add[:]  # only the provided lines

        # strip accidental empties and ensure final newline
        out = [o for o in out if o.strip()]

        with open(fpath, "w", newline="\n") as f:
            for o in out:
                f.write(o + "\n")
        total += 1
        print("wrote", fpath)

    print(f"Done. Updated {total} files in {args.labels_dir}")

if __name__ == "__main__":
    main()
