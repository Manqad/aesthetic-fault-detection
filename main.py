# main.py — Train/Test driver for PatchCore
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from train import train_patchcore
from test import detect_anomalies, evaluate_detection, visualize_after_testing
import test as test_mod  # for toggles like SAVED_CATEGORIES if needed

# --------------------------------------------------------------
# DEFAULT PATHS (edit if your layout changes)
# --------------------------------------------------------------
BASE_DIR        = Path("d:/fyp/aesthetic-fault-detection/Updated Dataset/acpart2")
MEMORY_BANK_DIR = Path("d:/fyp/aesthetic-fault-detection/models")
ANNOTATIONS_DIR = Path("d:/fyp/aesthetic-fault-detection/annotations")

DEFAULT_ANGLES = ["0º"]
DEFAULT_SIDES  = ["back"]            # e.g., ["front","back"]
DEFAULT_CATS   = ["bad1","bad2","bad3"]

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
    # Make angle tag robust for XML names (strip degree sign)
    angle_tag = str(angle).replace("º", "").replace("°", "")

    mem_path = MEMORY_BANK_DIR / f"patchcore_{angle}_{side}.pth"
    if not mem_path.exists():
        raise FileNotFoundError(f"Memory bank not found: {mem_path}")

    xml_file = f"{angle_tag}-{side}-bad-{cat}.xml"
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
        metrics = evaluate_detection(
            pred_boxes, gt_boxes, iou_thr=iou_thr, debug=verbose, img_name=name
        )

        # Be resilient to either "count IoU metrics" or "area metrics"
        # (some test.py variants return area_* keys)
        # Normalize keys so later aggregation is simple.
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
# PROCESS ONE ANGLE+SIDE (optional training -> testing)
# --------------------------------------------------------------
def process(angle: str, side: str, cats, do_train: bool, iou_thr: float, verbose: bool, viz: bool):
    if do_train:
        good_dir = BASE_DIR / angle / side / "good"
        mem_path = MEMORY_BANK_DIR / f"patchcore_{angle}_{side}.pth"
        if verbose:
            print(f"\nTraining memory bank for {angle} - {side}")
            print(f"  Good dir: {good_dir}")
            print(f"  Out:      {mem_path}")
        train_patchcore(str(good_dir), str(mem_path))

    # Reset one-time comparison saving per angle/side test run
    test_mod.SAVED_CATEGORIES = {c: False for c in cats}

    results = {}
    for cat in cats:
        if verbose:
            print(f"\nTesting {angle} - {side} - {cat}")
        results[cat] = test_category(angle, side, cat, iou_thr=iou_thr, verbose=verbose)

    # Optional: save one visualization per category
    if viz:
        visualize_after_testing(
            angle=angle,
            side=side,
            base_dir=BASE_DIR,
            annotations_dir=ANNOTATIONS_DIR,
            memory_bank_dir=MEMORY_BANK_DIR,
            categories=cats
        )

    return results

# --------------------------------------------------------------
# SUMMARY PRINTER
# --------------------------------------------------------------
def print_summary(all_results, quiet: bool):
    # all_results: {angle: {side: {cat: {imgname: {precision, recall, f1_score, is_area}}}}}
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
                # If any image used area metrics, tag this row as AREA
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
    p = argparse.ArgumentParser(description="PatchCore train/test driver")
    p.add_argument("--angles", nargs="+", default=DEFAULT_ANGLES,
                   help="Angles to process (e.g. 0º 45º 90º)")
    p.add_argument("--sides",  nargs="+", default=DEFAULT_SIDES,
                   help="Sides to process (e.g. front back)")
    p.add_argument("--cats",   nargs="+", default=DEFAULT_CATS,
                   help="Bad categories to test (e.g. bad1 bad2 bad3)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--train-and-test", action="store_true",
                      help="Run training first, then test (default mode if neither flag is given).")
    mode.add_argument("--test-only", action="store_true",
                      help="Skip training; only run testing using existing memory banks.")

    p.add_argument("--iou-thr", type=float, default=0.5,
                   help="IoU threshold used by evaluate_detection for TP matching (only for IoU-based eval).")
    p.add_argument("--viz", action="store_true",
                   help="Save exactly one side-by-side visualization per category.")
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

    # Decide whether to train
    do_train = True
    if args.test_only:
        do_train = False
    elif args.train_and_test:
        do_train = True
    # else: default to train+test

    all_results = {}
    for ang in args.angles:
        ang_results = {}
        for side in args.sides:
            res = process(
                angle=ang,
                side=side,
                cats=args.cats,
                do_train=do_train,
                iou_thr=args.iou_thr,
                verbose=verbose,
                viz=args.viz
            )
            ang_results[side] = res
        all_results[ang] = ang_results

    print_summary(all_results, quiet=args.quiet)

if __name__ == "__main__":
    main()
