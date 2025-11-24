# ==========================================================
# OPTIMIZED BINARY EVAL WITH EMERGENCY SAVING
# ==========================================================

from pathlib import Path
import csv
import xml.etree.ElementTree as ET
from functools import lru_cache
import pickle
import signal
import sys

from tqdm import tqdm
import numpy as np

import test as test_mod
from test import detect_anomalies


# ==========================================================
# USER CONFIG (unchanged)
# ==========================================================

BASE_DIR        = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/acpart2")
MEMORY_BANK_DIR = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/models")
ANNOTATIONS_DIR = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/annotations")

ANGLES   = ["0", "45upright"]
SIDES    = ["front", "back", "side2"]
BAD_CATS = ["bad1", "bad2", "bad3", "badgood"]

RUN_SWEEP = False

SWEEP_THR_MIN = 3.0
SWEEP_THR_MAX = 7.0
SWEEP_STEPS   = 40
SWEEP_SAMPLE_LIMIT = 0

SINGLE_EVAL_THRESHOLD = None
SINGLE_SAMPLE_LIMIT   = 0
SINGLE_OUT_CSV        = "binary_full_confusion.csv"
QUIET = False


# ==========================================================
# EMERGENCY SAVE VARIABLES
# ==========================================================

current_best = None
completed_results = []
PROGRESS_FILE = "sweep_progress.pkl"


# ==========================================================
# EMERGENCY SAVE HANDLER
# ==========================================================

def emergency_save(sig=None, frame=None):
    """Save progress on interrupt and print current best"""
    print(f"\n{'!'*60}")
    print("⚡ INTERRUPT RECEIVED! Saving current progress...")
    print(f"{'!'*60}")
    
    if completed_results:
        print(f"📊 Completed {len(completed_results)}/{SWEEP_STEPS} thresholds")
        
        # Print current best
        if current_best:
            thr, f1, res = current_best
            print(f"🏆 CURRENT BEST:")
            print(f"   Threshold: {thr:.3f}")
            print(f"   F1-Score:  {f1:.4f}")
            print(f"   MCC:       {res['metrics']['mcc']:.4f}")
            print(f"   Accuracy:  {res['metrics']['accuracy']:.4f}")
            print(f"   Precision: {res['metrics']['precision']:.4f}")
            print(f"   Recall:    {res['metrics']['recall']:.4f}")
        
        # Save all results
        save_data = {
            'current_best': current_best,
            'completed_results': completed_results,
            'sweep_config': {
                'thr_min': SWEEP_THR_MIN,
                'thr_max': SWEEP_THR_MAX,
                'steps': SWEEP_STEPS
            }
        }
        
        with open(PROGRESS_FILE, "wb") as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Progress saved to: {PROGRESS_FILE}")
        print("🔌 You can resume later or analyze the saved results")
        
    else:
        print("❌ No results completed yet")
    
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, emergency_save)


# ==========================================================
# FASTER XML LOADING (cached)
# ==========================================================

@lru_cache(maxsize=None)
def load_xml(xml_path_str):
    """Cached XML parser."""
    xml_path = Path(xml_path_str)
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


# ==========================================================
# PRELOAD ALL DATA (paths + XML) ONCE
# ==========================================================

def preload_dataset(sample_limit=0, quiet=False):
    """
    Preload:
    - All XML annotations (cached)
    - All image paths
    - All memory bank paths

    Returned structure is reused for every threshold.
    """
    dataset = []

    for ang in ANGLES:
        for side in SIDES:

            mem_path = MEMORY_BANK_DIR / f"patchcore_{ang}_{side}.pth"
            if not mem_path.exists():
                if not quiet:
                    print(f"[SKIP] No memory bank: {mem_path}")
                continue

            for cat in BAD_CATS:

                xml_path = ANNOTATIONS_DIR / f"{ang}-{side}-bad-{cat}.xml"
                img_dir  = BASE_DIR / ang / side / "bad" / cat

                if not xml_path.exists() or not img_dir.exists():
                    if not quiet:
                        print(f"[SKIP] Missing XML or folder for {ang}-{side}-{cat}")
                    continue

                ann = load_xml(str(xml_path))
                items = list(ann.items())
                if sample_limit > 0:
                    items = items[:sample_limit]

                dataset.append((ang, side, cat, mem_path, img_dir, items))

    return dataset


# ==========================================================
# METRICS (unchanged)
# ==========================================================

def metric_from_confusion(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0

    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced = 0.5 * (rec + tnr)

    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": tnr,
        "balanced_accuracy": balanced,
        "mcc": mcc,
    }


# ==========================================================
# CORE EVAL (now much faster)
# ==========================================================

def evaluate_binary_fast(dataset, threshold=None, out_csv="", quiet=False):
    if threshold is not None:
        test_mod.ANOMALY_THRESHOLD = float(threshold)

    test_mod.SAVED_CATEGORIES = {"bad1": True, "bad2": True, "bad3": True}

    TP = FP = FN = TN = 0
    rows = []

    for (ang, side, cat, mem_path, img_dir, items) in tqdm(dataset, disable=quiet):
        for name, gt_boxes in items:

            img_path = img_dir / name
            if not img_path.exists():
                continue

            _, pred_boxes, _ = detect_anomalies(
                str(img_path), str(mem_path), gt_boxes, cat=cat
            )

            gt_fault = len(gt_boxes) > 0
            pred_fault = len(pred_boxes) > 0

            if gt_fault:
                TP += pred_fault
                FN += (not pred_fault)
            else:
                FP += pred_fault
                TN += (not pred_fault)

            rows.append({
                "angle": ang,
                "side": side,
                "cat": cat,
                "image": name,
                "gt_faulty": int(gt_fault),
                "pred_faulty": int(pred_fault)
            })

    metrics = metric_from_confusion(TP, FP, FN, TN)

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        if not quiet:
            print(f"[WRITE] → {out_csv}")

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "metrics": metrics,
        "threshold": threshold,
        "rows": rows
    }


# ==========================================================
# SWEEP WITH EMERGENCY SAVING
# ==========================================================

def sweep_threshold_binary_fast(dataset, thr_min, thr_max, steps, quiet=False):
    global current_best, completed_results
    
    thresholds = np.linspace(thr_min, thr_max, steps)
    
    # Try to load previous progress
    try:
        with open(PROGRESS_FILE, "rb") as f:
            saved_data = pickle.load(f)
            completed_results = saved_data['completed_results']
            current_best = saved_data['current_best']
            print(f"🔄 RESUMED: {len(completed_results)}/{steps} thresholds already completed!")
            
            if current_best:
                thr, f1, res = current_best
                print(f"🏆 Previous best: thr={thr:.3f}, F1={f1:.4f}, MCC={res['metrics']['mcc']:.4f}")
            
    except FileNotFoundError:
        completed_results = []
        current_best = None
        print("🆕 Starting fresh sweep...")
    
    # Find where to start
    start_idx = len(completed_results)
    if start_idx >= len(thresholds):
        print("✅ All thresholds already completed! Returning results.")
        return current_best, completed_results
    
    print(f"🎯 Starting from threshold {start_idx + 1}/{len(thresholds)}")
    
    best = current_best
    
    # Continue from where we left off
    for i, thr in enumerate(thresholds[start_idx:], start_idx):
        try:
            res = evaluate_binary_fast(dataset, threshold=thr, quiet=True)
            f1 = res["metrics"]["f1"]
            
            # Store this result
            result_data = {
                "threshold": thr,
                "metrics": res["metrics"],
                "confusion": {"TP": res["TP"], "FP": res["FP"], "FN": res["FN"], "TN": res["TN"]}
            }
            completed_results.append(result_data)
            
            # Update best
            if (best is None) or (f1 > best[1]):
                best = (thr, f1, res)
                current_best = best
            
            # Print progress
            print(f"✅ {i+1}/{len(thresholds)}: thr={thr:.3f} → "
                  f"F1={f1:.4f}, MCC={res['metrics']['mcc']:.4f}, "
                  f"Acc={res['metrics']['accuracy']:.4f}")
            
            # Auto-save every 3 thresholds
            if (i + 1) % 3 == 0 or (i + 1) == len(thresholds):
                save_data = {
                    'current_best': current_best,
                    'completed_results': completed_results,
                    'sweep_config': {
                        'thr_min': thr_min,
                        'thr_max': thr_max,
                        'steps': steps
                    }
                }
                with open(PROGRESS_FILE, "wb") as f:
                    pickle.dump(save_data, f)
                print(f"💾 Auto-saved at {i+1}/{len(thresholds)}")
                
        except Exception as e:
            print(f"❌ Error at threshold {thr:.3f}: {e}")
            continue

    # Clean up progress file if completed
    if len(completed_results) == len(thresholds):
        try:
            Path(PROGRESS_FILE).unlink()
            print("🧹 Cleaned up progress file (completed)")
        except:
            pass
    
    return best, completed_results


# ==========================================================
# ANALYSIS FUNCTIONS
# ==========================================================

def analyze_saved_results():
    """Analyze saved results without running the sweep"""
    try:
        with open(PROGRESS_FILE, "rb") as f:
            data = pickle.load(f)
        
        print(f"\n{'='*60}")
        print("📊 SAVED RESULTS ANALYSIS")
        print(f"{'='*60}")
        print(f"Completed: {len(data['completed_results'])} thresholds")
        
        if data['current_best']:
            thr, f1, res = data['current_best']
            print(f"\n🏆 BEST RESULT:")
            print(f"   Threshold: {thr:.3f}")
            for metric, value in res['metrics'].items():
                print(f"   {metric.capitalize():<18}: {value:.4f}")
        
        print(f"\n📈 All Results (F1 & MCC):")
        for result in data['completed_results']:
            print(f"   thr={result['threshold']:.3f} → "
                  f"F1={result['metrics']['f1']:.4f}, "
                  f"MCC={result['metrics']['mcc']:.4f}")
                  
    except FileNotFoundError:
        print("❌ No saved progress file found")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    # Check if user wants to just analyze saved results
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_saved_results()
        sys.exit(0)
    
    dataset = preload_dataset(
        sample_limit=SWEEP_SAMPLE_LIMIT if RUN_SWEEP else SINGLE_SAMPLE_LIMIT,
        quiet=QUIET
    )

    if RUN_SWEEP:
        print("🚀 Starting threshold sweep with emergency saving...")
        print("💡 Press Ctrl+C at any time to save progress and exit!")
        print("💡 Run 'python script.py analyze' to view saved results without running\n")
        
        best, all_results = sweep_threshold_binary_fast(
            dataset,
            thr_min=SWEEP_THR_MIN,
            thr_max=SWEEP_THR_MAX,
            steps=SWEEP_STEPS,
            quiet=QUIET
        )
        
        if best:
            best_thr, best_f1, best_res = best
            print(f"\n🎉 FINAL BEST THRESHOLD: {best_thr:.3f}")
            print("📊 Final Metrics:")
            for metric, value in best_res["metrics"].items():
                print(f"   {metric.capitalize():<18}: {value:.4f}")

    else:
        res = evaluate_binary_fast(
            dataset,
            threshold=SINGLE_EVAL_THRESHOLD,
            out_csv=SINGLE_OUT_CSV,
            quiet=QUIET
        )
        print(res["metrics"])