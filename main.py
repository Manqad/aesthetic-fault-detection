# main.py — Train OR Test driver for PatchCore (no angles/sides in CLI)
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from train import train_patchcore
from test import detect_anomalies, evaluate_detection, visualize_after_testing
import test as test_mod  # for toggles like SAVED_CATEGORIES if needed

# --------------------------------------------------------------
# FIXED PATHS / DATA SCOPE (edit here if your layout changes)
# --------------------------------------------------------------
BASE_DIR        = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/acpart2")
MEMORY_BANK_DIR = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/models")
ANNOTATIONS_DIR = Path("C:/Users/mr08456/Desktop/fyp/aesthetic-fault-detection/annotations")

ANGLES = ["0","45upright"]                    # degree sign removed everywhere
SIDES  = ["front","back","side2"]               # e.g., ["front","back"]
BAD_CATS = ["bad1", "bad2", "bad3", "badgood"]

# --------------------------------------------------------------
# XML LOADER
# --------------------------------------------------------------
def load_xml(xml_path: Path):
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
            boxes.append([xtl, ytl, xbr - xtl, ybr - ytl])  # (x,y,w,h)
        ann[name] = boxes
    return ann

# --------------------------------------------------------------
# TEST ONE CATEGORY
# --------------------------------------------------------------
def test_category(angle: str, side: str, cat: str, iou_thr: float, verbose: bool):
    mem_path = MEMORY_BANK_DIR / f"patchcore_{angle}_{side}.pth"
    if not mem_path.exists():
        if verbose:
            print(f"[SKIP] Memory bank not found: {mem_path}")
        return {}

    xml_file = f"{angle}-{side}-bad-{cat}.xml"  # angle already clean (no º/°)
    xml_path = ANNOTATIONS_DIR / xml_file
    if not xml_path.exists():
        if verbose:
            print(f"[SKIP] Annotation missing: {xml_path}")
        return {}

    ann = load_xml(xml_path)
    img_dir = BASE_DIR / angle / side / "bad" / cat

    cat_res = {}
    for name, gt_boxes in ann.items():
        img_path = img_dir / name
        if not img_path.exists():
            if verbose:
                print(f"  [WARN] Missing image: {img_path}")
            continue

        # detect + evaluate
        _, pred_boxes, _ = detect_anomalies(
            str(img_path), str(mem_path), gt_boxes, cat
        )

        from PIL import Image
        with Image.open(img_path) as im:
            orig_w, orig_h = im.size


        metrics = evaluate_detection(
        pred_boxes,
        gt_boxes,
        iou_thr=iou_thr,
        debug=verbose,
        img_name=name,
        image_size=(orig_h, orig_w),   # if you track it earlier
        alpha=0.5                      # equal IoU/Area weight
        )


        # Normalize keys so later aggregation is simple
        if "precision" in metrics:
            cat_res[name] = {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "is_area": False,
            }
        else:
            cat_res[name] = {
                "precision": metrics.get("area_precision", 0.0),
                "recall": metrics.get("area_recall", 0.0),
                "f1_score": metrics.get("area_f1", 0.0),
                "is_area": True,
            }

        if verbose:
            label = "AREA" if cat_res[name]["is_area"] else "IOU"
            print(f"  {name} [{label}] → "
                  f"P:{cat_res[name]['precision']:.2f} "
                  f"R:{cat_res[name]['recall']:.2f} "
                  f"F1:{cat_res[name]['f1_score']:.2f}")

    return cat_res

# --------------------------------------------------------------
# TRAIN ALL (angles x sides)
# --------------------------------------------------------------
def train_all(verbose: bool):
    for ang in ANGLES:
        for side in SIDES:
            good_dir = BASE_DIR / ang / side / "good"
            mem_path = MEMORY_BANK_DIR / f"patchcore_{ang}_{side}.pth"
            if verbose:
                print("\n" + "="*60)
                print(f"TRAINING: angle={ang} side={side}")
                print("="*60)
                print(f"  Good dir: {good_dir}")
                print(f"  Out:      {mem_path}")
            train_patchcore(str(good_dir), str(mem_path))

# --------------------------------------------------------------
# TEST ALL (angles x sides x cats)
# --------------------------------------------------------------
def test_all(iou_thr: float, verbose: bool, viz: bool, quiet: bool):
    all_results = {}
    for ang in ANGLES:
        ang_results = {}
        for side in SIDES:
            # Reset one-time comparison saving per side run
            test_mod.SAVED_CATEGORIES = {c: False for c in BAD_CATS}

            if verbose:
                print("\n" + "="*60)
                print(f"TESTING: angle={ang} side={side}")
                print("="*60)

            res = {}
            for cat in BAD_CATS:
                if verbose:
                    print(f"\nTesting {ang} - {side} - {cat}")
                res[cat] = test_category(
                    ang, side, cat, iou_thr=iou_thr, verbose=verbose
                )

            # Optional: save one visualization per category
            if viz:
                visualize_after_testing(
                    angle=ang,
                    side=side,
                    base_dir=BASE_DIR,
                    annotations_dir=ANNOTATIONS_DIR,
                    memory_bank_dir=MEMORY_BANK_DIR,
                    categories=BAD_CATS
                )

            ang_results[side] = res
        all_results[ang] = ang_results

    print_summary(all_results, quiet=quiet)

# --------------------------------------------------------------
# SUMMARY PRINTER
# --------------------------------------------------------------
def print_summary(all_results, quiet: bool):
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for ang, sides in all_results.items():
        print(f"\n{ang}:")
        for side, cats in sides.items():
            for cat, imgs in cats.items():
                if not imgs:
                    continue
                ps = [m["precision"] for m in imgs.values()]
                rs = [m["recall"]    for m in imgs.values()]
                fs = [m["f1_score"]  for m in imgs.values()]
                used_area = any(m.get("is_area", False) for m in imgs.values())
                tag = "AREA" if used_area else "IOU"
                print(f"  {side} - {cat} [{tag}] → Avg P:{(sum(ps)/len(ps)):.2f} "
                      f"R:{(sum(rs)/len(rs)):.2f} F1:{(sum(fs)/len(fs)):.2f}")

    if not quiet:
        print("\n(Use --quiet to suppress per-image logs; this summary is always shown.)")

# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="PatchCore: train OR test (angles/sides are fixed in this file)."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true",
                      help="Train memory banks only (no testing).")
    mode.add_argument("--test", action="store_true",
                      help="Test only using existing memory banks.")

    p.add_argument("--iou-thr", type=float, default=0.2,
                   help="IoU threshold for TP matching in IoU-based eval.")
    p.add_argument("--viz", action="store_true",
                   help="(Testing only) Save one side-by-side visualization per category.")
    p.add_argument("--quiet", action="store_true",
                   help="Only print the final averages at the end.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-image evaluation details.")
    return p.parse_args()

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    args = parse_args()
    verbose = bool(args.verbose) and not args.quiet
    if args.quiet:
        verbose = False

    if args.train:
        train_all(verbose=verbose)
        return

    if args.test:
        test_all(
            iou_thr=args.iou_thr,
            verbose=verbose,
            viz=args.viz,
            quiet=args.quiet
        )
        return

if __name__ == "__main__":
    main()
