# binary_eval.py — Faulty-only binary evaluation (+ optional threshold sweep)
# Uses existing detect_anomalies() from test.py (detection unchanged).

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
SIDES    = ["front","back", "side2"]
BAD_CATS = ["bad1", "bad2", "bad3"]

# --- Mode selection ---
RUN_SWEEP       = True   # True  = sweep thresholds
                        # False = single evaluation at current test.ANOMALY_THRESHOLD

# --- Sweep config (used only if RUN_SWEEP = True) ---
SWEEP_THR_MIN   = 3.0
SWEEP_THR_MAX   = 7.0
SWEEP_STEPS     = 25     # number of thresholds between min and max
SWEEP_SAMPLE_LIMIT = 0   # 0 = use all images, >0 = cap per (angle,side,cat)

# --- Single-eval config (used if RUN_SWEEP = False) ---
SINGLE_EVAL_THRESHOLD = None  # None -> use current test_mod.ANOMALY_THRESHOLD
SINGLE_SAMPLE_LIMIT   = 0     # 0 = all
SINGLE_OUT_CSV        = "binary_faulty_only.csv"
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

def metrics_from_counts(tp, fn):
    """Faulty-only metrics: TP = detected faulty images, FN = missed."""
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    miss = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"recall": rec, "miss_rate": miss, "accuracy": rec}

# ======================================================
# CORE EVALUATION (one pass)
# ======================================================

def evaluate_faulty_only(threshold=None,
                         sample_limit=0,
                         out_csv="",
                         quiet=False):
    """
    Evaluate only BAD images.

    threshold:
        If not None, sets test_mod.ANOMALY_THRESHOLD before running.
        If None, uses whatever is already set in test.py.
    """
    if threshold is not None:
        test_mod.ANOMALY_THRESHOLD = float(threshold)

    # Don't spam comparison images while evaluating
    test_mod.SAVED_CATEGORIES = {"bad1": True, "bad2": True, "bad3": True}

    TP = 0
    FN = 0
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

                    if gt_faulty:
                        total_images += 1
                        if pred_faulty:
                            TP += 1
                        else:
                            FN += 1

                        rows.append({
                            "angle": ang,
                            "side": side,
                            "cat": cat,
                            "image": name,
                            "gt_faulty": int(gt_faulty),
                            "pred_faulty": int(pred_faulty)
                        })

    m = metrics_from_counts(TP, FN)

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["angle","side","cat","image","gt_faulty","pred_faulty"]
            )
        # reopen to actually write rows
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["angle","side","cat","image","gt_faulty","pred_faulty"]
            )
            writer.writeheader()
            writer.writerows(rows)
        if not quiet:
            print(f"\n[WRITE] per-image results → {out_csv}")

    if not quiet:
        print("\n=== Faulty-Only Evaluation ===")
        print(f"Threshold (ANOMALY_THRESHOLD): {test_mod.ANOMALY_THRESHOLD:.4f}")
        print(f"Images evaluated (faulty only): {total_images}")
        print(f"Detected correctly (TP): {TP}")
        print(f"Missed (FN): {FN}")
        print("\nMetrics:")
        print(f"  Recall (Detection Rate): {m['recall']:.4f}")
        print(f"  Miss Rate             : {m['miss_rate']:.4f}")

    return {
        "TP": TP,
        "FN": FN,
        "metrics": m,
        "n": total_images,
        "threshold": test_mod.ANOMALY_THRESHOLD,
    }

# ======================================================
# THRESHOLD SWEEP
# ======================================================

def sweep_threshold_faulty(thr_min, thr_max, steps,
                           sample_limit=0,
                           quiet=False):
    """
    Sweeps anomaly thresholds and finds the one with best recall
    (faulty-only detection).
    """
    thresholds = np.linspace(thr_min, thr_max, steps).tolist()
    results = []

    for t in thresholds:
        res = evaluate_faulty_only(
            threshold=t,
            sample_limit=sample_limit,
            out_csv="",  # no CSV per threshold
            quiet=True
        )
        recall = res["metrics"]["recall"]
        miss   = res["metrics"]["miss_rate"]
        results.append((t, recall, miss, res))

        if not quiet:
            print(f"THR={t:6.3f} → Recall={recall:.4f}  Miss={miss:.4f}  n={res['n']}")

    if not results:
        if not quiet:
            print("No results computed (empty dataset?)")
        return None, None

    best_t, best_rec, best_miss, best_res = max(results, key=lambda x: x[1])

    if not quiet:
        print("\n================= BEST FAULTY-ONLY THRESHOLD =================")
        print(f"Best Threshold : {best_t:.4f}")
        print(f"Recall@best    : {best_rec:.4f}")
        print(f"MissRate@best  : {best_miss:.4f}")
        print(f"TP={best_res['TP']}  FN={best_res['FN']}  n={best_res['n']}")
        print("===============================================================")

    return best_t, best_res

# ======================================================
# MAIN ENTRY (no CLI flags)
# ======================================================

if __name__ == "__main__":
    if RUN_SWEEP:
        # Find best threshold using faulty-only recall
        sweep_threshold_faulty(
            thr_min=SWEEP_THR_MIN,
            thr_max=SWEEP_THR_MAX,
            steps=SWEEP_STEPS,
            sample_limit=SWEEP_SAMPLE_LIMIT,
            quiet=QUIET,
        )
    else:
        # Single evaluation at a fixed threshold (or current one in test.py)
        evaluate_faulty_only(
            threshold=SINGLE_EVAL_THRESHOLD,
            sample_limit=SINGLE_SAMPLE_LIMIT,
            out_csv=SINGLE_OUT_CSV,
            quiet=QUIET,
        )
