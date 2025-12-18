# ==========================================================
# BINARY EVAL WITH COMPARISON IMAGES SAVING
# ==========================================================

from pathlib import Path
import csv
import xml.etree.ElementTree as ET
from functools import lru_cache
import pickle
import signal
import sys
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from tqdm import tqdm

import test as test_mod
from test import detect_anomalies


# ==========================================================
# USER CONFIG
# ==========================================================

BASE_DIR        = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/acpart2")
MEMORY_BANK_DIR = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/models")
ANNOTATIONS_DIR = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/annotations")
OUTPUT_DIR      = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/comparison_images")

ANGLES   = ["0", "45upright"]
SIDES    = ["front", "back", "side2"]
BAD_CATS = ["bad1", "bad2", "bad3", "badgood"]

# Fixed threshold for evaluation
FIXED_THRESHOLD = 4.25
SAMPLE_LIMIT   = 0
OUTPUT_CSV     = "binary_full_confusion.csv"
QUIET = False


# ==========================================================
# EMERGENCY SAVE HANDLER (kept for consistency)
# ==========================================================

def emergency_save(sig=None, frame=None):
    """Save progress on interrupt"""
    print(f"\n{'!'*60}")
    print("INTERRUPT RECEIVED!")
    print(f"{'!'*60}")
    print("Evaluation interrupted. Partial results may be in CSV.")
    sys.exit(0)

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
# CREATE COMPARISON IMAGES
# ==========================================================

def create_comparison_image(img_path, pred_boxes, gt_boxes, score_map, prediction, ground_truth):
    """Create a comparison image showing original, ground truth, and prediction"""
    
    # Read original image
    original_img = cv2.imread(str(img_path))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Create copies for annotations
    gt_img = original_img.copy()
    pred_img = original_img.copy()
    
    # Draw ground truth boxes (green)
    for box in gt_boxes:
        x, y, w, h = box
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green
    
    # Draw prediction boxes (red)
    for box in pred_boxes:
        x, y, w, h = box
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Red
    
    # Create heatmap from score_map (normalize to 0-255)
    if score_map is not None:
        heatmap = cv2.resize(score_map, (original_img.shape[1], original_img.shape[0]))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255
        heatmap = heatmap.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend heatmap with original image
        alpha = 0.5
        heatmap_blend = cv2.addWeighted(original_img, 1-alpha, heatmap, alpha, 0)
    else:
        heatmap_blend = original_img
    
    # Create the comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground Truth
    axes[0, 1].imshow(gt_img)
    gt_status = "FAULTY" if ground_truth else "GOOD"
    axes[0, 1].set_title(f'Ground Truth: {gt_status} ({len(gt_boxes)} boxes)', 
                         fontsize=14, fontweight='bold', color='green')
    axes[0, 1].axis('off')
    
    # Prediction
    axes[1, 0].imshow(pred_img)
    pred_status = "FAULTY" if prediction else "GOOD"
    pred_color = 'red' if prediction else 'blue'
    axes[1, 0].set_title(f'Prediction: {pred_status} ({len(pred_boxes)} boxes)', 
                         fontsize=14, fontweight='bold', color=pred_color)
    axes[1, 0].axis('off')
    
    # Heatmap
    axes[1, 1].imshow(heatmap_blend)
    axes[1, 1].set_title('Anomaly Heatmap', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add overall result annotation (using ASCII characters)
    result_text = "MATCH" if ground_truth == prediction else "MISMATCH"
    result_symbol = "[OK]" if ground_truth == prediction else "[X]"
    result_color = "green" if ground_truth == prediction else "red"
    fig.suptitle(f'Comparison: {result_symbol} {result_text} | GT: {gt_status} vs Pred: {pred_status}', 
                 fontsize=16, fontweight='bold', color=result_color)
    
    plt.tight_layout()
    
    return fig


def save_comparison_image(img_path, ang, side, cat, pred_boxes, gt_boxes, score_map, 
                         prediction, ground_truth, output_dir):
    """Create and save comparison image"""
    
    # Create directory structure based on match type
    match_type = "matches" if ground_truth == prediction else "mismatches"
    if not ground_truth and not prediction:
        error_type = "true_negative"
    elif ground_truth and prediction:
        error_type = "true_positive"
    elif not ground_truth and prediction:
        error_type = "false_positive"
    else:  # ground_truth and not prediction
        error_type = "false_negative"
    
    # Create directory: comparison_images/angle/side/match_type/error_type/category/
    save_dir = output_dir / ang / side / match_type / error_type / cat
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison figure
    fig = create_comparison_image(img_path, pred_boxes, gt_boxes, score_map, 
                                 prediction, ground_truth)
    
    # Save the figure
    original_name = img_path.stem
    save_path = save_dir / f"{original_name}_comparison.png"
    
    try:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # Close the figure to free memory
        return save_path
    except Exception as e:
        print(f"Failed to save comparison image {save_path}: {e}")
        plt.close(fig)
        return None


# ==========================================================
# CORE EVAL WITH COMPARISON IMAGES SAVING
# ==========================================================

def evaluate_binary_with_comparisons(dataset, threshold=None, out_csv="", output_dir=None, quiet=False):
    if threshold is not None:
        test_mod.ANOMALY_THRESHOLD = float(threshold)

    test_mod.SAVED_CATEGORIES = {"bad1": True, "bad2": True, "bad3": True}

    TP = FP = FN = TN = 0
    rows = []
    comparison_count = 0
    mismatch_count = 0

    # Create output directory if it doesn't exist
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Comparison images will be saved to: {output_dir}")

    for (ang, side, cat, mem_path, img_dir, items) in tqdm(dataset, disable=quiet):
        for name, gt_boxes in items:

            img_path = img_dir / name
            if not img_path.exists():
                continue

            _, pred_boxes, score_map = detect_anomalies(
                str(img_path), str(mem_path), gt_boxes, cat=cat
            )

            gt_fault = len(gt_boxes) > 0
            pred_fault = len(pred_boxes) > 0

            # Count confusion matrix
            if gt_fault:
                TP += pred_fault
                FN += (not pred_fault)
            else:
                FP += pred_fault
                TN += (not pred_fault)

            # Save comparison images for ALL cases
            if output_dir:
                saved_path = save_comparison_image(
                    img_path, ang, side, cat, pred_boxes, gt_boxes, score_map,
                    pred_fault, gt_fault, output_dir
                )
                if saved_path:
                    comparison_count += 1
                    if gt_fault != pred_fault:
                        mismatch_count += 1

            rows.append({
                "angle": ang,
                "side": side,
                "cat": cat,
                "image": name,
                "gt_faulty": int(gt_fault),
                "pred_faulty": int(pred_fault),
                "match": int(gt_fault == pred_fault),
                "gt_boxes": len(gt_boxes),
                "pred_boxes": len(pred_boxes)
            })

    metrics = metric_from_confusion(TP, FP, FN, TN)

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        if not quiet:
            print(f"[WRITE] -> {out_csv}")

    # Print summary
    if output_dir and comparison_count > 0:
        print(f"Saved {comparison_count} comparison images to: {output_dir}")
        print(f"   Matches: {comparison_count - mismatch_count}")
        print(f"   Mismatches: {mismatch_count}")
        print(f"   - True Positives: {TP}")
        print(f"   - False Positives: {FP}")
        print(f"   - False Negatives: {FN}")
        print(f"   - True Negatives: {TN}")

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "metrics": metrics,
        "threshold": threshold,
        "rows": rows,
        "comparison_count": comparison_count,
        "mismatch_count": mismatch_count
    }


# ==========================================================
# ANALYSIS FUNCTIONS
# ==========================================================

def analyze_comparisons(comparison_dir):
    """Analyze the saved comparison images"""
    comparison_dir = Path(comparison_dir)
    
    if not comparison_dir.exists():
        print("No comparison images directory found")
        return
    
    print(f"\n{'='*60}")
    print("COMPARISON IMAGES ANALYSIS")
    print(f"{'='*60}")
    
    # Count by match type and error type
    for match_type in ["matches", "mismatches"]:
        match_dir = comparison_dir / "**" / "**" / match_type  # ang/side/match_type
        match_images = list(comparison_dir.rglob(f"*/{match_type}/*/*/*.png"))
        
        if match_images:
            print(f"\n{match_type.upper()}: {len(match_images)} images")
            
            # Count by error type within match type
            for error_type in ["true_positive", "true_negative", "false_positive", "false_negative"]:
                error_images = [img for img in match_images if img.parent.name == error_type]
                if error_images:
                    print(f"   {error_type.replace('_', ' ').title()}: {len(error_images)}")
                    
                    # Count by angle
                    angles = set(img.parent.parent.parent.parent.name for img in error_images)
                    for ang in sorted(angles):
                        ang_count = len([img for img in error_images if img.parent.parent.parent.parent.name == ang])
                        print(f"     {ang}: {ang_count}")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    # Check if user wants to analyze comparisons
    if len(sys.argv) > 1 and sys.argv[1] == "analyze_comparisons":
        analyze_comparisons(OUTPUT_DIR)
        sys.exit(0)
    
    print("Starting binary evaluation with comparison images...")
    print(f"Using threshold: {FIXED_THRESHOLD}")
    print(f"Comparison images will be saved to: {OUTPUT_DIR}")
    print("Press Ctrl+C to interrupt evaluation\n")
    
    dataset = preload_dataset(
        sample_limit=SAMPLE_LIMIT,
        quiet=QUIET
    )

    res = evaluate_binary_with_comparisons(
        dataset,
        threshold=FIXED_THRESHOLD,
        out_csv=OUTPUT_CSV,
        output_dir=OUTPUT_DIR,
        quiet=QUIET
    )
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Threshold: {FIXED_THRESHOLD}")
    print(f"Total Images: {len(res['rows'])}")
    print(f"Comparison Images Saved: {res['comparison_count']}")
    print(f"Mismatched Images: {res['mismatch_count']}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {res['TP']}")
    print(f"  False Positives: {res['FP']}") 
    print(f"  False Negatives: {res['FN']}")
    print(f"  True Negatives:  {res['TN']}")
    print(f"\nMetrics:")
    for metric, value in res["metrics"].items():
        print(f"  {metric.capitalize():<18}: {value:.4f}")
    
    if res['comparison_count'] > 0:
        print(f"\nRun 'python script.py analyze_comparisons' to see detailed breakdown")
