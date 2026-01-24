# crop_margin_dataset.py
import argparse, glob, os, cv2

def process(split_img, split_lbl, out_img, out_lbl, margin=0.10, min_area_frac=1e-3):
    os.makedirs(out_img, exist_ok=True); os.makedirs(out_lbl, exist_ok=True)
    for imgp in glob.glob(os.path.join(split_img, "*.jpg")) + glob.glob(os.path.join(split_img, "*.png")):
        base = os.path.splitext(os.path.basename(imgp))[0]
        lblp = os.path.join(split_lbl, base + ".txt")
        im = cv2.imread(imgp);  h, w = im.shape[:2]
        L = int(margin * w); R = w - L; T = int(margin * h); B = h - T
        crop = im[T:B, L:R];  H2, W2 = crop.shape[:2]

        # write image
        cv2.imwrite(os.path.join(out_img, base + ".jpg"), crop)

        # rewrite labels if present
        out_lines = []
        if os.path.exists(lblp):
            with open(lblp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    cls, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                    # denormalize
                    x = xc * w; y = yc * h; ww = bw * w; hh = bh * h
                    # keep only boxes whose center falls inside the crop
                    if not (L <= x < R and T <= y < B): 
                        continue
                    # shift to crop coords
                    x -= L; y -= T
                    # clip to crop bounds
                    x1 = max(0, x - ww/2); y1 = max(0, y - hh/2)
                    x2 = min(W2, x + ww/2); y2 = min(H2, y + hh/2)
                    if x2 <= x1 or y2 <= y1: 
                        continue
                    # re-center and re-normalize
                    xn = ((x1 + x2) / 2) / W2
                    yn = ((y1 + y2) / 2) / H2
                    wn = (x2 - x1) / W2
                    hn = (y2 - y1) / H2
                    # drop tiny boxes if you want
                    if wn * hn < min_area_frac:
                        continue
                    out_lines.append(f"{cls} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}")

        with open(os.path.join(out_lbl, base + ".txt"), "w") as f:
            f.write("\n".join(out_lines))
    print(f"Done: {split_img} → {out_img}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs-train", required=True)
    ap.add_argument("--labels-train", required=True)
    ap.add_argument("--imgs-val", required=True)
    ap.add_argument("--labels-val", required=True)
    ap.add_argument("--out-imgs-train", required=True)
    ap.add_argument("--out-labels-train", required=True)
    ap.add_argument("--out-imgs-val", required=True)
    ap.add_argument("--out-labels-val", required=True)
    ap.add_argument("--margin", type=float, default=0.10)
    args = ap.parse_args()
    process(args.imgs_train, args.labels_train, args.out_imgs_train, args.out_labels_train, args.margin)
    process(args.imgs_val,   args.labels_val,   args.out_imgs_val,   args.out_labels_val,   args.margin)
