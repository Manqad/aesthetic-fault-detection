# tune_thresholds_overall.py — sweep anomaly thresholds and show F1 evolution live
# Shows per-threshold F1s, saves CSV, and plots curves at the end.

import argparse
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import csv
import matplotlib.pyplot as plt

import test as test_mod
from test import (
    detect_anomalies,
    evaluate_detection,
    evaluate_detection_area,
    evaluate_detection_group,
)

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
BASE_DIR        = Path("D:/fyp/aesthetic-fault-detection/acpart2")
MEMORY_BANK_DIR = Path("D:/fyp/aesthetic-fault-detection/models")
ANNOTATIONS_DIR = Path("D:/fyp/aesthetic-fault-detection/annotations")
ANGLES = ["0"]
SIDES  = ["front", "back", "side2"]
BAD_CATS = ["bad1", "bad2", "bad3"]

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def load_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ann = {}
    for img in root.findall("image"):
        name = img.get("name")
        boxes = []
        for box in img.findall("box"):
            xtl = float(box.get("xtl")); ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr")); ybr = float(box.get("ybr"))
            boxes.append([xtl, ytl, xbr - xtl, ybr - ytl])
        ann[name] = boxes
    return ann


def eval_all(pred_boxes, gt_boxes, score_map, iou_thr=0.2, group_eps=0.05):
    """Evaluate using all three metrics."""
    H, W = score_map.shape[:2]
    m_iou   = evaluate_detection(pred_boxes, gt_boxes, iou_thr=iou_thr, debug=False)
    m_area  = evaluate_detection_area(pred_boxes, gt_boxes, image_size=(H, W), debug=False)
    m_group = evaluate_detection_group(pred_boxes, gt_boxes, image_size=(H, W), iou_eps=group_eps, debug=False)
    return {
        "iou_f1": m_iou["f1_score"],
        "area_f1": m_area["area_f1"],
        "group_f1": m_group["group_f1"],
        "overall": None,  # will fill later
    }


# -------------------------------------------------------
# SWEEP
# -------------------------------------------------------
def sweep_global(thr_min, thr_max, steps, iou_thr, group_eps, w_iou, w_area, w_group, sample_limit,
                 out_csv, out_png):

    thresholds = np.linspace(thr_min, thr_max, steps).tolist()
    print(f"\n🔍 Sweeping {len(thresholds)} thresholds from {thr_min:.2f} → {thr_max:.2f}")
    print(f"Weights → IoU={w_iou:.2f}, Area={w_area:.2f}, Group={w_group:.2f}\n")

    rows = []
    for t in thresholds:
        test_mod.SAVED_CATEGORIES = {"bad1": True, "bad2": True, "bad3": True}
        test_mod.ANOMALY_THRESHOLD = float(t)

        iou_f1s, area_f1s, group_f1s = [], [], []

        for ang in ANGLES:
            for side in SIDES:
                mem_path = MEMORY_BANK_DIR / f"patchcore_{ang}_{side}.pth"
                if not mem_path.exists():
                    continue

                for cat in BAD_CATS:
                    xml_file = f"{ang}-{side}-bad-{cat}.xml"
                    xml_path = ANNOTATIONS_DIR / xml_file
                    img_dir  = BASE_DIR / ang / side / "bad" / cat
                    if not xml_path.exists() or not img_dir.exists():
                        continue

                    ann = load_xml(xml_path)
                    ct = 0
                    for name, gt_boxes in ann.items():
                        img_path = img_dir / name
                        if not img_path.exists():
                            continue
                        if sample_limit and ct >= sample_limit:
                            break
                        ct += 1

                        _, pred_boxes, score_map = detect_anomalies(
                            str(img_path), str(mem_path), gt_boxes, cat
                        )
                        res = eval_all(pred_boxes, gt_boxes, score_map,
                                       iou_thr=iou_thr, group_eps=group_eps)
                        iou_f1s.append(res["iou_f1"])
                        area_f1s.append(res["area_f1"])
                        group_f1s.append(res["group_f1"])

        # Mean across all samples
        if len(iou_f1s) == 0:
            continue
        iou_f1 = np.mean(iou_f1s)
        area_f1 = np.mean(area_f1s)
        group_f1 = np.mean(group_f1s)
        overall = w_iou * iou_f1 + w_area * area_f1 + w_group * group_f1

        rows.append({
            "threshold": t,
            "iou_f1": iou_f1,
            "area_f1": area_f1,
            "group_f1": group_f1,
            "overall": overall,
            "n_images": len(iou_f1s)
        })

        print(f"THR={t:5.2f} → IoU-F1={iou_f1:6.3f} | Area-F1={area_f1:6.3f} | "
              f"Group-F1={group_f1:6.3f} | Overall={overall:6.3f} ({len(iou_f1s)} imgs)")

    if not rows:
        print("No valid results. Check paths/annotations.")
        return

    # Sort by threshold
    rows = sorted(rows, key=lambda r: r["threshold"])
    best = max(rows, key=lambda r: r["overall"])

    # Save CSV
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n📁 CSV saved → {out_csv}")

    # Plot curves
    xs = [r["threshold"] for r in rows]
    y_over = [r["overall"] for r in rows]
    y_iou = [r["iou_f1"] for r in rows]
    y_area = [r["area_f1"] for r in rows]
    y_group = [r["group_f1"] for r in rows]

    plt.figure(figsize=(7,5))
    plt.plot(xs, y_over, label="Overall-F1", linewidth=2)
    plt.plot(xs, y_iou, label="IoU-F1", linestyle="--")
    plt.plot(xs, y_area, label="Area-F1", linestyle="--")
    plt.plot(xs, y_group, label="Group-F1", linestyle="--")
    plt.xlabel("Anomaly Threshold")
    plt.ylabel("Average F1")
    plt.title("F1 vs Anomaly Threshold")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        print(f"🖼️  Plot saved → {out_png}")
    plt.show()

    print("\n================== BEST THRESHOLD ==================")
    print(f"Best Threshold : {best['threshold']:.3f}")
    print(f"Overall-F1     : {best['overall']:.3f}")
    print(f"IoU-F1         : {best['iou_f1']:.3f}")
    print(f"Area-F1        : {best['area_f1']:.3f}")
    print(f"Group-F1       : {best['group_f1']:.3f}")
    print(f"Samples used   : {best['n_images']}")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Threshold sweep + F1 visualization")
    p.add_argument("--thr-min", type=float, default=0.5)
    p.add_argument("--thr-max", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--iou-thr", type=float, default=0.2)
    p.add_argument("--group-eps", type=float, default=0.05)
    p.add_argument("--w-iou", type=float, default=0.34)
    p.add_argument("--w-area", type=float, default=0.33)
    p.add_argument("--w-group", type=float, default=0.33)
    p.add_argument("--sample-limit", type=int, default=0)
    p.add_argument("--out-csv", default="threshold_sweep.csv")
    p.add_argument("--out-png", default="threshold_sweep.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    s = args.w_iou + args.w_area + args.w_group
    w_iou, w_area, w_group = args.w_iou / s, args.w_area / s, args.w_group / s

    sweep_global(
        thr_min=args.thr_min,
        thr_max=args.thr_max,
        steps=args.steps,
        iou_thr=args.iou_thr,
        group_eps=args.group_eps,
        w_iou=w_iou, w_area=w_area, w_group=w_group,
        sample_limit=args.sample_limit,
        out_csv=args.out_csv,
        out_png=args.out_png
    )
