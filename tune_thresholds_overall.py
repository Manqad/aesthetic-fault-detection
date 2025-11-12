# tune_thresholds_overall.py — one-pass sweep that optimizes a single Overall-F1
# Overall-F1 = w_iou * IoU-F1 + w_area * Area-F1 + w_group * Group-F1
# (Detection logic remains unchanged; we only vary ANOMALY_THRESHOLD.)

import argparse
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

import test as test_mod
from test import (
    detect_anomalies,
    evaluate_detection,          # IoU metric
    evaluate_detection_area,     # Area metric
    evaluate_detection_group,    # Group metric
)

# ---- COPY these from your main.py if needed ----
BASE_DIR        = Path("D:/fyp/aesthetic-fault-detection/acpart2")
MEMORY_BANK_DIR = Path("D:/fyp/aesthetic-fault-detection/models")
ANNOTATIONS_DIR = Path("D:/fyp/aesthetic-fault-detection/annotations")
ANGLES = ["0"]
SIDES  = ["front", "back", "side2"]
BAD_CATS = ["bad1", "bad2", "bad3"]
# -----------------------------------------------

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
            boxes.append([xtl, ytl, xbr - xtl, ybr - ytl])  # (x,y,w,h)
        ann[name] = boxes
    return ann

def eval_all(pred_boxes, gt_boxes, score_map, iou_thr=0.2, group_eps=0.05):
    """Return (iou_f1, area_f1, group_f1, iou_p, iou_r, area_p, area_r, group_p, group_r)."""
    H, W = score_map.shape[:2]

    m_iou   = evaluate_detection(pred_boxes, gt_boxes, iou_thr=iou_thr, debug=False)
    iou_f1  = float(m_iou["f1_score"]);  iou_p = float(m_iou["precision"]);  iou_r = float(m_iou["recall"])

    m_area  = evaluate_detection_area(pred_boxes, gt_boxes, image_size=(H, W), debug=False)
    area_f1 = float(m_area["area_f1"]);  area_p = float(m_area["area_precision"]); area_r = float(m_area["area_recall"])

    m_group = evaluate_detection_group(pred_boxes, gt_boxes, image_size=(H, W), iou_eps=group_eps, debug=False)
    group_f1 = float(m_group["group_f1"]); group_p = float(m_group["group_precision"]); group_r = float(m_group["group_recall"])

    return iou_f1, area_f1, group_f1, iou_p, iou_r, area_p, area_r, group_p, group_r

def sweep_global(thr_min, thr_max, steps, iou_thr, group_eps, w_iou, w_area, w_group, sample_limit):
    thresholds = np.linspace(thr_min, thr_max, steps).tolist()
    print(f"[INFO] Sweeping {len(thresholds)} thresholds from {thr_min} to {thr_max}")
    print(f"[INFO] Weights → w_iou={w_iou:.2f}, w_area={w_area:.2f}, w_group={w_group:.2f}  (sum={w_iou+w_area+w_group:.2f})")

    # Per-threshold aggregates
    agg = {
        "overall": {t: [] for t in thresholds},
        "iou_f1":  {t: [] for t in thresholds},
        "area_f1": {t: [] for t in thresholds},
        "group_f1":{t: [] for t in thresholds},
        "iou_p":   {t: [] for t in thresholds},
        "iou_r":   {t: [] for t in thresholds},
        "area_p":  {t: [] for t in thresholds},
        "area_r":  {t: [] for t in thresholds},
        "group_p": {t: [] for t in thresholds},
        "group_r": {t: [] for t in thresholds},
    }

    for t in thresholds:
        test_mod.ANOMALY_THRESHOLD = float(t)

        for ang in ANGLES:
            for side in SIDES:
                mem_path = MEMORY_BANK_DIR / f"patchcore_{ang}_{side}.pth"
                if not mem_path.exists():
                    print(f"[SKIP] Missing memory bank: {mem_path}")
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

                        # detection unchanged; only threshold differs
                        _, pred_boxes, score_map = detect_anomalies(str(img_path), str(mem_path), gt_boxes, cat)

                        iou_f1, area_f1, group_f1, iou_p, iou_r, area_p, area_r, group_p, group_r = \
                            eval_all(pred_boxes, gt_boxes, score_map, iou_thr=iou_thr, group_eps=group_eps)

                        overall = w_iou * iou_f1 + w_area * area_f1 + w_group * group_f1

                        agg["overall"][t].append(overall)
                        agg["iou_f1"][t].append(iou_f1)
                        agg["area_f1"][t].append(area_f1)
                        agg["group_f1"][t].append(group_f1)
                        agg["iou_p"][t].append(iou_p)
                        agg["iou_r"][t].append(iou_r)
                        agg["area_p"][t].append(area_p)
                        agg["area_r"][t].append(area_r)
                        agg["group_p"][t].append(group_p)
                        agg["group_r"][t].append(group_r)

    # pick best by mean Overall-F1
    best_t, best_score = None, -1.0
    for t in thresholds:
        if not agg["overall"][t]:
            continue
        mean_overall = float(np.mean(agg["overall"][t]))
        if mean_overall > best_score:
            best_score = mean_overall
            best_t = t

    if best_t is None:
        print("[ERROR] No results. Check paths/annotations.")
        return

    def mean_of(key): return float(np.mean(agg[key][best_t])) if agg[key][best_t] else 0.0

    print("\n================== GLOBAL BEST THRESHOLD ==================")
    print(f"Best Threshold : {best_t:.4f}")
    print(f"Overall-F1@best (w_iou={w_iou:.2f}, w_area={w_area:.2f}, w_group={w_group:.2f}) = {best_score:.4f}")
    print("Breakdown @best:")
    print(f"  IoU-F1   : {mean_of('iou_f1'):.4f}")
    print(f"  Area-F1  : {mean_of('area_f1'):.4f}")
    print(f"  Group-F1 : {mean_of('group_f1'):.4f}")
    print(f"  IoU  P/R : {mean_of('iou_p'):.4f} / {mean_of('iou_r'):.4f}")
    print(f"  Area P/R : {mean_of('area_p'):.4f} / {mean_of('area_r'):.4f}")
    print(f"  GroupP/R : {mean_of('group_p'):.4f} / {mean_of('group_r'):.4f}")

    # also show a small leaderboard of thresholds
    leaderboard = []
    for t in thresholds:
        if agg["overall"][t]:
            leaderboard.append((t, float(np.mean(agg["overall"][t]))))
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    print("\nTop thresholds by Overall-F1:")
    for t, s in leaderboard[:5]:
        print(f"  thr={t:.4f}  Overall-F1={s:.4f}")

def parse_args():
    p = argparse.ArgumentParser("Global threshold sweep (single Overall-F1 = weighted IoU/Area/Group)")
    p.add_argument("--thr-min", type=float, default=0.5)
    p.add_argument("--thr-max", type=float, default=8.0)
    p.add_argument("--steps",   type=int,   default=30)
    p.add_argument("--iou-thr", type=float, default=0.2, help="Match threshold for IoU evaluator")
    p.add_argument("--group-eps", type=float, default=0.05, help="Overlap eps for building Group components (mask IoU)")
    p.add_argument("--w-iou",   type=float, default=0.34, help="Weight for IoU-F1 in Overall-F1")
    p.add_argument("--w-area",  type=float, default=0.33, help="Weight for Area-F1 in Overall-F1")
    p.add_argument("--w-group", type=float, default=0.33, help="Weight for Group-F1 in Overall-F1")
    p.add_argument("--sample-limit", type=int, default=0, help="Limit images per (angle,side,cat) for speed (0=all)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Normalize weights if user passed arbitrary values
    s = args.w_iou + args.w_area + args.w_group
    if s <= 0:
        print("[ERROR] Sum of weights must be > 0")
        raise SystemExit(1)
    w_iou   = args.w_iou   / s
    w_area  = args.w_area  / s
    w_group = args.w_group / s

    sweep_global(
        thr_min=args.thr_min, thr_max=args.thr_max, steps=args.steps,
        iou_thr=args.iou_thr, group_eps=args.group_eps,
        w_iou=w_iou, w_area=w_area, w_group=w_group,
        sample_limit=args.sample_limit
    )
