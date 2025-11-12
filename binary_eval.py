# binary_eval_faulty_only.py — Evaluate only on bad parts (faulty images)
# Now with progress bar (tqdm) to show evaluation progress.

import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm  # <-- progress bar
from test import detect_anomalies

# ----------------------------------------------------------
# CONFIG — adjust to match your setup
# ----------------------------------------------------------
BASE_DIR        = Path("D:/fyp/aesthetic-fault-detection/acpart2")
MEMORY_BANK_DIR = Path("D:/fyp/aesthetic-fault-detection/models")
ANNOTATIONS_DIR = Path("D:/fyp/aesthetic-fault-detection/annotations")

ANGLES = ["0"]
SIDES  = ["front", "back", "side2"]
BAD_CATS = ["bad1", "bad2", "bad3"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def load_xml(xml_path: Path):
    """Load CVAT XML → dict: name -> [x,y,w,h] boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ann = {}
    for img in root.findall("image"):
        name = img.get("name")
        boxes = []
        for box in img.findall("box"):
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            boxes.append([xtl, ytl, xbr - xtl, ybr - ytl])
        ann[name] = boxes
    return ann


def metrics_from_counts(tp, fn):
    """Since we only test faulty images: TP = correctly detected, FN = missed."""
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"recall": recall, "miss_rate": miss_rate, "accuracy": recall}


# ----------------------------------------------------------
# Evaluation logic
# ----------------------------------------------------------
def evaluate_faulty_only(sample_limit=0, out_csv="binary_faulty_results.csv", quiet=False):
    """
    Evaluate only on 'bad' images (ignore good ones).
    Each image: if GT has ≥1 box → it's faulty; if prediction has ≥1 box → correctly detected.
    """
    TP = 0
    FN = 0
    total = 0
    rows = []

    for ang in ANGLES:
        for side in SIDES:
            mem_path = MEMORY_BANK_DIR / f"patchcore_{ang}_{side}.pth"
            if not mem_path.exists():
                if not quiet:
                    print(f"[SKIP] Missing memory bank: {mem_path}")
                continue

            for cat in BAD_CATS:
                xml_file = f"{ang}-{side}-bad-{cat}.xml"
                xml_path = ANNOTATIONS_DIR / xml_file
                img_dir  = BASE_DIR / ang / side / "bad" / cat
                if not xml_path.exists() or not img_dir.exists():
                    if not quiet:
                        print(f"[SKIP] Missing XML or images for {cat}")
                    continue

                ann = load_xml(xml_path)
                img_items = list(ann.items())
                if sample_limit > 0:
                    img_items = img_items[:sample_limit]

                print(f"\n[PROCESS] {ang}-{side}-{cat}: {len(img_items)} images")

                # tqdm progress bar for each category
                for name, gt_boxes in tqdm(img_items, desc=f"  Evaluating {cat}", unit="img"):
                    img_path = img_dir / name
                    if not img_path.exists():
                        if not quiet:
                            tqdm.write(f"  [WARN] Missing image: {img_path}")
                        continue

                    _, pred_boxes, _ = detect_anomalies(str(img_path), str(mem_path), gt_boxes, cat)

                    gt_faulty   = (len(gt_boxes) > 0)
                    pred_faulty = (len(pred_boxes) > 0)

                    if gt_faulty:
                        total += 1
                        if pred_faulty:
                            TP += 1
                        else:
                            FN += 1
                        rows.append({
                            "angle": ang,
                            "side": side,
                            "cat": cat,
                            "image": name,
                            "detected": int(pred_faulty)
                        })

    # Metrics
    m = metrics_from_counts(TP, FN)
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["angle","side","cat","image","detected"])
            writer.writeheader()
            writer.writerows(rows)
        if not quiet:
            print(f"[WRITE] Results saved → {out_csv}")

    # Summary
    if not quiet:
        print("\n=== Faulty-Only Evaluation ===")
        print(f"Images evaluated (faulty only): {total}")
        print(f"Detected correctly (TP): {TP}")
        print(f"Missed (FN): {FN}")
        print("\nMetrics:")
        print(f"  Recall (Detection Rate): {m['recall']:.4f}")
        print(f"  Miss Rate             : {m['miss_rate']:.4f}")
        print(f"  Accuracy (same as recall): {m['accuracy']:.4f}")

    return {"TP": TP, "FN": FN, "metrics": m, "n": total}


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Binary evaluation — only faulty images (with progress bar)")
    p.add_argument("--sample-limit", type=int, default=0, help="Limit images per category (0=all)")
    p.add_argument("--out-csv", default="binary_faulty_results.csv", help="CSV output path")
    p.add_argument("--quiet", action="store_true", help="Suppress extra logs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_faulty_only(sample_limit=args.sample_limit, out_csv=args.out_csv, quiet=args.quiet)
