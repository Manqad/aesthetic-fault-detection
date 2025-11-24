# tune_thresholds_overall.py — sweep anomaly thresholds and show F1 evolution live
# Shows per-threshold F1s, saves CSV, and plots curves at the end.

import argparse
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

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
BASE_DIR        = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/acpart2")
MEMORY_BANK_DIR = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/models")
ANNOTATIONS_DIR = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/annotations")
ANGLES = ["0", "45"]
SIDES  = ["front", "back", "side2"]
BAD_CATS = ["bad1", "bad2", "bad3", "badgood"]

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
        "iou_precision": m_iou["precision"],
        "iou_recall": m_iou["recall"],
        "area_f1": m_area["area_f1"],
        "area_precision": m_area["area_precision"], 
        "area_recall": m_area["area_recall"],
        "group_f1": m_group["group_f1"],
        "group_precision": m_group["group_precision"],
        "group_recall": m_group["group_recall"],
        "overall": None,  # will fill later
    }


# -------------------------------------------------------
# SWEEP WITH LIVE VISUALIZATION
# -------------------------------------------------------
def sweep_global(thr_min, thr_max, steps, iou_thr, group_eps, w_iou, w_area, w_group, sample_limit,
                 out_csv, out_png):

    thresholds = np.linspace(thr_min, thr_max, steps).tolist()
    print(f"\n🔍 Sweeping {len(thresholds)} thresholds from {thr_min:.2f} → {thr_max:.2f}")
    print(f"Weights → IoU={w_iou:.2f}, Area={w_area:.2f}, Group={w_group:.2f}\n")

    # Initialize live plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.suptitle("Live Threshold Sweep Progress", fontsize=14, fontweight='bold')
    
    # Storage for all results
    all_results = []
    rows = []
    
    # Progress bar for thresholds
    pbar = tqdm(thresholds, desc="Threshold Sweep", unit="thr")
    
    for t in pbar:
        test_mod.SAVED_CATEGORIES = {"bad1": True, "bad2": True, "bad3": True}
        test_mod.ANOMALY_THRESHOLD = float(t)

        # Initialize metrics storage for this threshold
        iou_f1s, iou_precisions, iou_recalls = [], [], []
        area_f1s, area_precisions, area_recalls = [], [], []
        group_f1s, group_precisions, group_recalls = [], [], []
        
        total_images = 0
        
        # Process each configuration
        config_pbar = tqdm(total=len(ANGLES)*len(SIDES)*len(BAD_CATS), 
                          desc=f"Configs (thr={t:.2f})", leave=False, unit="cfg")
        
        for ang in ANGLES:
            for side in SIDES:
                mem_path = MEMORY_BANK_DIR / f"patchcore_{ang}_{side}.pth"
                if not mem_path.exists():
                    config_pbar.update(1)
                    continue

                for cat in BAD_CATS:
                    xml_file = f"{ang}-{side}-bad-{cat}.xml"
                    xml_path = ANNOTATIONS_DIR / xml_file
                    img_dir  = BASE_DIR / ang / side / "bad" / cat
                    if not xml_path.exists() or not img_dir.exists():
                        config_pbar.update(1)
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
                        
                        # Store all metrics
                        iou_f1s.append(res["iou_f1"])
                        iou_precisions.append(res["iou_precision"])
                        iou_recalls.append(res["iou_recall"])
                        area_f1s.append(res["area_f1"])
                        area_precisions.append(res["area_precision"])
                        area_recalls.append(res["area_recall"])
                        group_f1s.append(res["group_f1"])
                        group_precisions.append(res["group_precision"])
                        group_recalls.append(res["group_recall"])
                        
                        total_images += 1
                    
                    config_pbar.update(1)
        
        config_pbar.close()

        # Calculate mean metrics for this threshold
        if len(iou_f1s) == 0:
            pbar.write(f"❌ THR={t:5.2f} → No valid images processed")
            continue
            
        iou_f1 = np.mean(iou_f1s)
        iou_precision = np.mean(iou_precisions)
        iou_recall = np.mean(iou_recalls)
        area_f1 = np.mean(area_f1s)
        area_precision = np.mean(area_precisions)
        area_recall = np.mean(area_recalls)
        group_f1 = np.mean(group_f1s)
        group_precision = np.mean(group_precisions)
        group_recall = np.mean(group_recalls)
        overall = w_iou * iou_f1 + w_area * area_f1 + w_group * group_f1

        # Store results
        result = {
            "threshold": t,
            "iou_f1": iou_f1,
            "iou_precision": iou_precision,
            "iou_recall": iou_recall,
            "area_f1": area_f1,
            "area_precision": area_precision,
            "area_recall": area_recall,
            "group_f1": group_f1,
            "group_precision": group_precision,
            "group_recall": group_recall,
            "overall": overall,
            "n_images": len(iou_f1s)
        }
        all_results.append(result)
        rows.append(result)

        # Update progress bar description
        pbar.set_description(f"THR={t:.2f} | Overall={overall:.3f} | Imgs={total_images}")

        # Print detailed metrics for this threshold
        pbar.write(f"\n{'='*80}")
        pbar.write(f"📊 THRESHOLD: {t:.3f} (Images: {total_images})")
        pbar.write(f"{'-'*80}")
        pbar.write(f"Overall Score: {overall:.4f}")
        pbar.write(f"{'-'*80}")
        pbar.write("IoU Metrics:")
        pbar.write(f"  F1: {iou_f1:.4f} | Precision: {iou_precision:.4f} | Recall: {iou_recall:.4f}")
        pbar.write("Area Metrics:")
        pbar.write(f"  F1: {area_f1:.4f} | Precision: {area_precision:.4f} | Recall: {area_recall:.4f}")
        pbar.write("Group Metrics:")
        pbar.write(f"  F1: {group_f1:.4f} | Precision: {group_precision:.4f} | Recall: {group_recall:.4f}")
        pbar.write(f"{'='*80}")

        # Update live plot
        update_live_plot(ax1, ax2, all_results, t)
        plt.pause(0.1)  # Small pause to update display

    pbar.close()
    plt.ioff()

    if not rows:
        print("❌ No valid results. Check paths/annotations.")
        return

    # Sort by threshold
    rows = sorted(rows, key=lambda r: r["threshold"])
    best = max(rows, key=lambda r: r["overall"])

    # Save CSV with all metrics
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n📁 CSV saved → {out_csv}")

    # Create final plot
    create_final_plot(rows, best, out_png)

    # Print final best results
    print_best_results(best)


def update_live_plot(ax1, ax2, results, current_thr):
    """Update the live plot with current progress"""
    xs = [r["threshold"] for r in results]
    
    # Clear axes
    ax1.clear()
    ax2.clear()
    
    # Plot 1: F1 Scores
    ax1.plot(xs, [r["overall"] for r in results], 'b-', linewidth=3, label="Overall-F1", marker='o')
    ax1.plot(xs, [r["iou_f1"] for r in results], 'r--', linewidth=2, label="IoU-F1")
    ax1.plot(xs, [r["area_f1"] for r in results], 'g--', linewidth=2, label="Area-F1")
    ax1.plot(xs, [r["group_f1"] for r in results], 'm--', linewidth=2, label="Group-F1")
    ax1.axvline(x=current_thr, color='orange', linestyle=':', alpha=0.7, label=f'Current: {current_thr:.2f}')
    ax1.set_xlabel("Anomaly Threshold")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("Live F1 Scores vs Threshold")
    ax1.legend()
    ax1.grid(True, linestyle="--", linewidth=0.6)
    
    # Plot 2: Precision and Recall
    ax2.plot(xs, [r["iou_precision"] for r in results], 'r-', linewidth=2, label="IoU-Precision")
    ax2.plot(xs, [r["iou_recall"] for r in results], 'r--', linewidth=2, label="IoU-Recall")
    ax2.plot(xs, [r["area_precision"] for r in results], 'g-', linewidth=2, label="Area-Precision")
    ax2.plot(xs, [r["area_recall"] for r in results], 'g--', linewidth=2, label="Area-Recall")
    ax2.plot(xs, [r["group_precision"] for r in results], 'm-', linewidth=2, label="Group-Precision")
    ax2.plot(xs, [r["group_recall"] for r in results], 'm--', linewidth=2, label="Group-Recall")
    ax2.axvline(x=current_thr, color='orange', linestyle=':', alpha=0.7)
    ax2.set_xlabel("Anomaly Threshold")
    ax2.set_ylabel("Precision / Recall")
    ax2.set_title("Live Precision & Recall vs Threshold")
    ax2.legend()
    ax2.grid(True, linestyle="--", linewidth=0.6)
    
    plt.tight_layout()


def create_final_plot(rows, best, out_png):
    """Create the final comprehensive plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    plt.suptitle("Final Threshold Analysis", fontsize=16, fontweight='bold')
    
    xs = [r["threshold"] for r in rows]
    
    # Plot 1: All F1 Scores
    ax1.plot(xs, [r["overall"] for r in rows], 'b-', linewidth=3, label="Overall-F1", marker='o')
    ax1.plot(xs, [r["iou_f1"] for r in rows], 'r--', linewidth=2, label="IoU-F1")
    ax1.plot(xs, [r["area_f1"] for r in rows], 'g--', linewidth=2, label="Area-F1")
    ax1.plot(xs, [r["group_f1"] for r in rows], 'm--', linewidth=2, label="Group-F1")
    ax1.axvline(x=best["threshold"], color='red', linestyle='-', alpha=0.8, 
                label=f'Best: {best["threshold"]:.3f}')
    ax1.set_xlabel("Anomaly Threshold")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1 Scores vs Threshold")
    ax1.legend()
    ax1.grid(True, linestyle="--", linewidth=0.6)
    
    # Plot 2: IoU Metrics
    ax2.plot(xs, [r["iou_f1"] for r in rows], 'r-', linewidth=2, label="IoU-F1")
    ax2.plot(xs, [r["iou_precision"] for r in rows], 'r--', linewidth=2, label="IoU-Precision")
    ax2.plot(xs, [r["iou_recall"] for r in rows], 'r:', linewidth=2, label="IoU-Recall")
    ax2.axvline(x=best["threshold"], color='red', linestyle='-', alpha=0.8)
    ax2.set_xlabel("Anomaly Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("IoU Metrics vs Threshold")
    ax2.legend()
    ax2.grid(True, linestyle="--", linewidth=0.6)
    
    # Plot 3: Area Metrics
    ax3.plot(xs, [r["area_f1"] for r in rows], 'g-', linewidth=2, label="Area-F1")
    ax3.plot(xs, [r["area_precision"] for r in rows], 'g--', linewidth=2, label="Area-Precision")
    ax3.plot(xs, [r["area_recall"] for r in rows], 'g:', linewidth=2, label="Area-Recall")
    ax3.axvline(x=best["threshold"], color='red', linestyle='-', alpha=0.8)
    ax3.set_xlabel("Anomaly Threshold")
    ax3.set_ylabel("Score")
    ax3.set_title("Area Metrics vs Threshold")
    ax3.legend()
    ax3.grid(True, linestyle="--", linewidth=0.6)
    
    # Plot 4: Group Metrics
    ax4.plot(xs, [r["group_f1"] for r in rows], 'm-', linewidth=2, label="Group-F1")
    ax4.plot(xs, [r["group_precision"] for r in rows], 'm--', linewidth=2, label="Group-Precision")
    ax4.plot(xs, [r["group_recall"] for r in rows], 'm:', linewidth=2, label="Group-Recall")
    ax4.axvline(x=best["threshold"], color='red', linestyle='-', alpha=0.8)
    ax4.set_xlabel("Anomaly Threshold")
    ax4.set_ylabel("Score")
    ax4.set_title("Group Metrics vs Threshold")
    ax4.legend()
    ax4.grid(True, linestyle="--", linewidth=0.6)
    
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"🖼️  Final plot saved → {out_png}")
    plt.show()


def print_best_results(best):
    """Print comprehensive best results"""
    print("\n" + "="*80)
    print("🏆 BEST THRESHOLD RESULTS")
    print("="*80)
    print(f"📈 Optimal Threshold: {best['threshold']:.4f}")
    print(f"⭐ Overall Score: {best['overall']:.4f}")
    print(f"📊 Images Used: {best['n_images']}")
    print("-"*80)
    print("IoU Detection Metrics:")
    print(f"  F1:       {best['iou_f1']:.4f}")
    print(f"  Precision:{best['iou_precision']:.4f}")
    print(f"  Recall:   {best['iou_recall']:.4f}")
    print("-"*80)
    print("Area-based Metrics:")
    print(f"  F1:       {best['area_f1']:.4f}")
    print(f"  Precision:{best['area_precision']:.4f}")
    print(f"  Recall:   {best['area_recall']:.4f}")
    print("-"*80)
    print("Group-based Metrics:")
    print(f"  F1:       {best['group_f1']:.4f}")
    print(f"  Precision:{best['group_precision']:.4f}")
    print(f"  Recall:   {best['group_recall']:.4f}")
    print("="*80)


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Threshold sweep + F1 visualization")
    p.add_argument("--thr-min", type=float, default=3.0)
    p.add_argument("--thr-max", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=25)
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