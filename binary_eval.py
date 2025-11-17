# binary_eval.py — Full binary evaluation (faulty vs good) + optional threshold sweep
# Uses your existing detect_anomalies() from test.py (detection unchanged).

from pathlib import Path
import csv
import xml.etree.ElementTree as ET

from tqdm import tqdm
import numpy as np

import test as test_mod
from test import detect_anomalies

# ======================================================
# USER CONFIG
# ======================================================

# Dataset / paths (same as main.py)
BASE_DIR        = Path("D:/fyp/aesthetic-fault-detection/acpart2")
MEMORY_BANK_DIR = Path("D:/fyp/aesthetic-fault-detection/models")
ANNOTATIONS_DIR = Path("D:/fyp/aesthetic-fault-detection/annotations")

ANGLES   = ["0"]
SIDES    = ["front", "back", "side2"]
BAD_CATS = ["bad1", "bad2", "bad3", "badgood"]
# Note: "badgood" XMLs should have NO <box> → treated as GT good images.

# --- Mode selection ---
RUN_SWEEP       = False   # True  = sweep thresholds, False = single eval

# --- Sweep config (used only if RUN_SWEEP = True) ---
SWEEP_THR_MIN       = 3.0
SWEEP_THR_MAX       = 7.0
SWEEP_STEPS         = 25     # number of thresholds between min and max
SWEEP_SAMPLE_LIMIT  = 0      # 0 = all images, >0 = cap per (angle,side,cat)

# --- Single-eval config (used if RUN_SWEEP = False) ---
SINGLE_EVAL_THRESHOLD = None  # None -> use current test_mod.ANOMALY_THRESHOLD
SINGLE_SAMPLE_LIMIT   = 0     # 0 = all
SINGLE_OUT_CSV        = "binary_full_confusion.csv"
QUIET = False                # True = less printing, no tqdm bars

# ======================================================
# HELPERS
# ======================================================

def load_xml(xml_path: Path):
    """Load CVAT-style XML: returns dict[name] = list of [x,y,w,h] boxes."""
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

def metrics_from_confusion(tp, fp, fn, tn):
    """
    Binary metrics. Positive class = 'faulty' (has at least one GT box).
    """
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / max(1, (tp + fp + fn + tn))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec
    }

# ======================================================
# CORE EVALUATION (one pass at a given threshold)
# ======================================================

def evaluate_binary(threshold=None,
                    sample_limit=0,
                    out_csv="",
                    quiet=False):
    """
    Full binary evaluation on all bad categories (including 'badgood').

    For each image:
      gt_faulty   = (len(gt_boxes) > 0)
      pred_faulty = (len(pred_boxes) > 0)

      TP: gt_faulty   & pred_faulty
      FN: gt_faulty   & not pred_faulty
      FP: not gt_faulty & pred_faulty
      TN: not gt_faulty & not pred_faulty
    """
    if threshold is not None:
        test_mod.ANOMALY_THRESHOLD = float(threshold)

    # Avoid comparison save spam during evaluation
    test_mod.SAVED_CATEGORIES = {"bad1": True, "bad2": True, "bad3": True}

    TP = FP = FN = TN = 0
    total_images = 0
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
                        print(f"[SKIP] Missing XML or images for {ang}-{side}-{cat}")
                    continue

                ann = load_xml(xml_path)
                img_items = list(ann.items())
                if sample_limit > 0:
                    img_items = img_items[:sample_limit]

                if not quiet:
                    print(f"\n[TEST] {ang}-{side}-{cat}: {len(img_items)} images  "
                          f"(thr={test_mod.ANOMALY_THRESHOLD:.3f})")

                for name, gt_boxes in tqdm(img_items,
                                           desc=f"{cat}",
                                           unit="img",
                                           disable=quiet):
                    img_path = img_dir / name
                    if not img_path.exists():
                        if not quiet:
                            tqdm.write(f"  [WARN] Missing image: {img_path}")
                        continue

                    # Detection logic from test.py (unchanged)
                    _, pred_boxes, _ = detect_anomalies(
                        str(img_path), str(mem_path), gt_boxes, cat=cat
                    )

                    gt_faulty   = (len(gt_boxes)  > 0)
                    pred_faulty = (len(pred_boxes) > 0)

                    # Full confusion matrix logic
                    if   gt_faulty and pred_faulty: TP += 1
                    elif gt_faulty and not pred_faulty: FN += 1
                    elif (not gt_faulty) and pred_faulty: FP += 1
                    else: TN += 1

                    total_images += 1
                    rows.append({
                        "angle": ang,
                        "side": side,
                        "cat": cat,
                        "image": name,
                        "gt_faulty": int(gt_faulty),
                        "pred_faulty": int(pred_faulty)
                    })

    m = metrics_from_confusion(TP, FP, FN, TN)

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["angle","side","cat","image","gt_faulty","pred_faulty"]
            )
            writer.writeheader()
            writer.writerows(rows)
        if not quiet:
            print(f"\n[WRITE] per-image binary results → {out_csv}")

    if not quiet:
        print("\n=== Full Binary Evaluation (faulty vs good) ===")
        print(f"Threshold (ANOMALY_THRESHOLD): {test_mod.ANOMALY_THRESHOLD:.4f}")
        print(f"Images evaluated: {total_images}")
        print("\nConfusion matrix (Positive = faulty):")
        print(f"  TP: {TP:5d}  FP: {FP:5d}")
        print(f"  FN: {FN:5d}  TN: {TN:5d}")
        print("\nMetrics:")
        print(f"  Accuracy   : {m['accuracy']:.4f}")
        print(f"  Precision  : {m['precision']:.4f}  (PPV)")
        print(f"  Recall     : {m['recall']:.4f}   (TPR / Sensitivity)")
        print(f"  Specificity: {m['specificity']:.4f}  (TNR)")
        print(f"  F1-score   : {m['f1']:.4f}")

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "metrics": m,
        "n": total_images,
        "threshold": test_mod.ANOMALY_THRESHOLD,
    }

# ======================================================
# THRESHOLD SWEEP (optional)
# ======================================================

def sweep_threshold_binary(thr_min, thr_max, steps,
                           sample_limit=0,
                           quiet=False):
    """
    Sweep anomaly thresholds and choose the best one w.r.t F1
    for binary classification (faulty vs good).
    """
    thresholds = np.linspace(thr_min, thr_max, steps).tolist()
    results = []

    for t in thresholds:
        res = evaluate_binary(
            threshold=t,
            sample_limit=sample_limit,
            out_csv="",      # don't write CSV per threshold
            quiet=True
        )
        m = res["metrics"]
        results.append((t, m["f1"], res))

        if not quiet:
            print(f"THR={t:6.3f} → F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  "
                  f"P={m['precision']:.4f} R={m['recall']:.4f}  "
                  f"Spec={m['specificity']:.4f}  n={res['n']}")

    if not results:
        if not quiet:
            print("No results computed (check dataset / paths).")
        return None, None

    best_t, best_f1, best_res = max(results, key=lambda x: x[1])

    if not quiet:
        print("\n================== BEST BINARY THRESHOLD ==================")
        print(f"Best Threshold : {best_t:.4f}")
        m = best_res["metrics"]
        print(f"F1@best        : {m['f1']:.4f}")
        print(f"Accuracy@best  : {m['accuracy']:.4f}")
        print(f"Precision@best : {m['precision']:.4f}")
        print(f"Recall@best    : {m['recall']:.4f}")
        print(f"Specificity@best: {m['specificity']:.4f}")
        print(f"TP={best_res['TP']} FP={best_res['FP']} FN={best_res['FN']} TN={best_res['TN']}  n={best_res['n']}")
        print("===========================================================")

    return best_t, best_res

# ======================================================
# MAIN ENTRY (no CLI flags)
# ======================================================

if __name__ == "__main__":
    if RUN_SWEEP:
        sweep_threshold_binary(
            thr_min=SWEEP_THR_MIN,
            thr_max=SWEEP_THR_MAX,
            steps=SWEEP_STEPS,
            sample_limit=SWEEP_SAMPLE_LIMIT,
            quiet=QUIET,
        )
    else:
        evaluate_binary(
            threshold=SINGLE_EVAL_THRESHOLD,
            sample_limit=SINGLE_SAMPLE_LIMIT,
            out_csv=SINGLE_OUT_CSV,
            quiet=QUIET,
        )
