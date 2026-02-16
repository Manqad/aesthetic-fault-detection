
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Optional for Green Screen (Background Removal)
try:
    from rembg import remove as rembg_remove
    _HAS_REMBG = True
except ImportError:
    _HAS_REMBG = False
    print("Warning: rembg not found. Background removal will be disabled.")

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
DATASET_ROOT = Path(r"c:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\dataset without bg")
RESULTS_FILE = Path("evaluation_results.txt")

# Configuration for THIS RUN
USE_BG_REMOVAL = False  # Disabled because dataset is already preprocessed
RUN_NAME = "Preprocessed Dataset (No BG) + 180px Mask"
# Training on Masked images -> Models should be saved to new dir
MODELS_DIR = Path(r"c:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\models_preprocessed_masks")
MISMATCH_DIR = Path(r"c:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\mismatches_preprocessed_masks")

IMG_SIZE = 1024
CROP_PX = 180 # Applied for masking border
ANOMALY_THRESHOLD = 4.25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MISMATCH_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR = Path(r"c:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\annotations_new_dataset")

# --------------------------------------------------------------
# UTILS
# --------------------------------------------------------------
def make_foreground_mask(pil_rgb: Image.Image) -> np.ndarray:
    # If the input ALREADY has background removed (black/white), we might not need a mask
    # The preprocessed dataset has BLACK background (r=0,g=0,b=0).
    # But evaluate_image converts to RGB.
    # So background becomes BLACK (0,0,0).
    
    # Simple Mask Generation from Image Content (Non-Black Pixels)
    img_np = np.array(pil_rgb)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return mask

def load_xml_boxes(xml_path):
    if not xml_path.exists():
        return {}
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

def compute_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "MCC": mcc
    }

def extract_all_results(filepath):
    """
    Parses 'evaluation_results.txt' to extract metrics for ALL runs found.
    Returns a dict: { "Run Name": { "Accuracy": ..., ... }, ... }
    """
    if not filepath.exists():
        return {}
        
    with open(filepath, "r") as f:
        content = f.read()
    
    results = {}
    
    # Parse Run 1 (Legacy)
    m1 = {}
    match1 = re.search(r"FINAL RESULTS\s*=+\s*(.*?)(?=\n\n|\Z)", content, re.DOTALL)
    if match1:
        block = match1.group(1)
        m1 = parse_metrics_block(block)
        if m1: results["Initial Run (Unknown BG)"] = m1
            
    # Parse Subsequent Runs
    # Format: RUN: Name\n---\nmetrics
    pattern = r"RUN:\s*(.*?)\n-+\n(.*?)(?=\n\n|\Z|={10})"
    matches = re.finditer(pattern, content, re.DOTALL)
    for m in matches:
        name = m.group(1).strip()
        block = m.group(2)
        metrics = parse_metrics_block(block)
        if metrics:
            results[name] = metrics
            
    return results

def parse_metrics_block(block):
    metrics = {}
    interesting_keys = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "MCC"]
    for line in block.splitlines():
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val_str = parts[1].strip()
                if key in interesting_keys:
                    try:
                        metrics[key] = float(val_str)
                    except ValueError:
                        pass
    return metrics

# --------------------------------------------------------------
# MODEL (PatchCore)
# --------------------------------------------------------------
class PatchCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = wide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        self.layer2 = nn.Sequential(*list(self.backbone.children())[:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[:7])
        self.memory_bank = None

    def extract_features(self, x):
        with torch.no_grad():
            x2 = self.layer2(x)
            x3 = self.layer3(x)
            # Resize x3 to match x2 spatial dims
            x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
            
            feats = []
            for feat in [x2, x3]:
                b, c, h, w = feat.shape
                # Flatten spatial dims: (B, C, H*W) -> permute (B, H*W, C)
                feat = feat.view(b, c, h*w).permute(0, 2, 1)
                feats.append(feat)
            # Concatenate along channel dimension C
            return torch.cat(feats, dim=2)

    def compute_anomaly_score_map(self, features, device, chunk=1024):
        if self.memory_bank is None:
            raise ValueError("Memory bank not loaded")
            
        bank = self.memory_bank.to(device)
        feat = features.view(-1, features.size(-1)).to(device)
        
        bank = bank.float()
        feat = feat.float()
        
        mins = []
        with torch.no_grad():
            for i in range(0, feat.size(0), chunk):
                f = feat[i:i+chunk]
                d = torch.cdist(f, bank, p=2)
                # Compute average distance to k nearest neighbors
                vals, _ = d.topk(k=9, dim=1, largest=False)
                mins.append(vals.mean(dim=1))
                
        # mins is list of (chunk_size, ), concat -> (Total_pixels, )
        return torch.cat(mins, dim=0)

# --------------------------------------------------------------
# TRAIN
# --------------------------------------------------------------
def train_model(good_dir, model_path):
    print(f"Training model for {good_dir} (MASKED BORDER)...")
    model = PatchCore().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    images = list(good_dir.glob("*.jpg")) + list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpeg"))
    if not images:
        print(f"Skipping training: No images in {good_dir}")
        return
        
    all_feats = []
    batch_size = 4
    
    for i in tqdm(range(0, len(images), batch_size), desc="Extracting Features"):
        batch_files = images[i:i+batch_size]
        batch_imgs = []
        for f in batch_files:
            pil_img = Image.open(f).convert("RGB")
            
            # 1. MASK BORDER (Same as testing)
            w, h = pil_img.size
            if CROP_PX > 0:
                img_np = np.array(pil_img)
                crop_mask = np.zeros((h, w), dtype=np.uint8)
                x1, y1 = CROP_PX, CROP_PX
                x2, y2 = w - CROP_PX, h - CROP_PX
                
                if x2 > x1 and y2 > y1:
                    crop_mask[y1:y2, x1:x2] = 1
                    img_np[crop_mask == 0] = 0
                    pil_img = Image.fromarray(img_np)
            
            # Simple transform
            batch_imgs.append(transform(pil_img))
            
        if not batch_imgs: continue
        batch_tensor = torch.stack(batch_imgs).to(device)
        feats = model.extract_features(batch_tensor)
        all_feats.append(feats.cpu())
        
    if not all_feats:
        print("No features extracted.")
        return

    all_feats = torch.cat(all_feats, dim=0)
    all_feats = all_feats.view(-1, all_feats.shape[-1])
    
    # Coreset Subsampling
    coreset_cap = 200000 
    if all_feats.shape[0] > coreset_cap:
        idx = torch.randperm(all_feats.shape[0])[:coreset_cap]
        all_feats = all_feats[idx]
        
    torch.save(all_feats, model_path)
    print(f"Saved model to {model_path}")

# --------------------------------------------------------------
# EVALUATE
# --------------------------------------------------------------
def evaluate_image(model, img_path, gt_boxes=None):
    pil_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    
    # 1. CROP (by MASKING the outer border)
    # Why masking? To preserve scale relative to training images (which are uncropped).
    # Physically cropping & resizing would zoom the image, creating a scale mismatch.
    if CROP_PX > 0:
        # Create a black image and paste the center crop
        img_np = np.array(pil_img)
        # Create mask: 0 for border, 1 for center
        crop_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        # Define ROI (Region of Interest)
        x1, y1 = CROP_PX, CROP_PX
        x2, y2 = orig_w - CROP_PX, orig_h - CROP_PX
        
        if x2 > x1 and y2 > y1:
            crop_mask[y1:y2, x1:x2] = 1
            # Apply mask to image (Black out border)
            img_np[crop_mask == 0] = 0
            pil_img = Image.fromarray(img_np)
        else:
            print(f"Warning: CROP_PX={CROP_PX} too large for image {orig_w}x{orig_h}")

    # 2. Background Mask
    mask = make_foreground_mask(pil_img)
    mask_bool = (mask > 0).astype(np.uint8)
    
    # 2. Preprocess (Black out BG)
    img_np = np.array(pil_img)
    img_with_mask = img_np.copy()
    img_with_mask[mask == 0] = 0 
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor = transform(Image.fromarray(img_with_mask)).unsqueeze(0).to(device)
    
    # Inference
    feats = model.extract_features(tensor)
    scores = model.compute_anomaly_score_map(feats, device).cpu().numpy()
    
    # Reshape & Resize
    N = len(scores)
    side = int(np.sqrt(N))
    score_map = scores.reshape(side, side)
    score_map = cv2.resize(score_map, (orig_w, orig_h))
    
    # Apply Mask to Score (Suppress background anomaly scores)
    masked_score = score_map.copy()
    masked_score[mask_bool == 0] = 0 
    
    max_score = masked_score.max()
    is_faulty = max_score > ANOMALY_THRESHOLD
    
    # Visualization Data
    vis_data = {
        "pil_img": pil_img,
        "masked_score": masked_score,
        "mask": mask,                 # Keep mask for visualization
        "bg_removed_img": img_with_mask, # Keep BG removed image
        "pred_boxes": [], 
        "gt_boxes": gt_boxes if gt_boxes else [],
        "max_score": max_score
    }
    
    if is_faulty:
        # Get Boxes from high-score regions
        binary = (masked_score > ANOMALY_THRESHOLD).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
            # Filter small noise
            if area >= 20: 
                vis_data["pred_boxes"].append([x, y, w, h])
                
    return is_faulty, vis_data

def save_visualization(vis_data, save_path):
    pil_img = vis_data["pil_img"]
    masked_score = vis_data["masked_score"]
    mask = vis_data["mask"]
    bg_removed_img = vis_data["bg_removed_img"]
    
    # 1. Original
    img_vis = np.array(pil_img)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    
    # 2. Mask (Gray -> BGR)
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 3. BG Removed (RGB -> BGR)
    bg_removed_vis = cv2.cvtColor(bg_removed_img, cv2.COLOR_RGB2BGR)

    # 4. Heatmap
    norm_score = np.clip(masked_score, 0, 15) / 15.0 * 255
    heatmap = cv2.applyColorMap(norm_score.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_vis, 0.6, heatmap, 0.4, 0)
    
    # 5. GT Boxes (Green)
    img_gt = img_vis.copy()
    for (x, y, x2, y2) in vis_data["gt_boxes"]: 
        w, h = x2, y2 
        cv2.rectangle(img_gt, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 5)
        
    # 6. Pred Boxes (Red)
    img_pred = img_vis.copy()
    for (x, y, w, h) in vis_data["pred_boxes"]:
        cv2.rectangle(img_pred, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 5)
        
    # Create 2x3 Grid (Rows x Cols)
    # Row 1: Original, Mask, BG Removed
    # Row 2: Heatmap, GT, Pred
    
    # Ensure all same size (should be)
    row1 = np.hstack([img_vis, mask_vis, bg_removed_vis])
    row2 = np.hstack([overlay, img_gt, img_pred])
    
    final = np.vstack([row1, row2])
    
    label = f"Max Score: {vis_data['max_score']:.2f}"
    cv2.putText(final, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.imwrite(str(save_path), final)

# --------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------
def main():
    print(f"Starting Evaluation: {RUN_NAME}...")
    
    # Global Metrics for CURRENT RUN
    TP = FP = FN = TN = 0
    
    # 1. Discover Views
    views = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        if "good1" in dirs:
            views.append(Path(root))
            
    print(f"Found {len(views)} views.")
    
    for view_dir in views:
        rel_path = view_dir.relative_to(DATASET_ROOT)
        view_key = str(rel_path).replace(os.sep, "_").replace(" ", "")
        print(f"\nProcessing View: {view_key}")
        
        # 1. Train / Load Model
        model_path = MODELS_DIR / f"{view_key}.pth"
        if not model_path.exists():
            train_model(view_dir / "good1", model_path)
        else:
            print(f"Model exists: {model_path}")
            
        try:
            model = PatchCore().to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.memory_bank = state_dict
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            continue
            
        # 2. Evaluate Good Test
        goodtest_dir = view_dir / "goodtest" / "val1"
        if goodtest_dir.exists():
            images = list(goodtest_dir.glob("*.jpg")) + list(goodtest_dir.glob("*.png"))
            for img_path in tqdm(images, desc="Eval Good"):
                is_faulty, vis_data = evaluate_image(model, img_path)
                
                if not is_faulty:
                    TN += 1
                else:
                    FP += 1
                    out_name = f"FP_{view_key}_{img_path.name}"
                    save_visualization(vis_data, MISMATCH_DIR / out_name)
                    
        # 3. Evaluate Bad Test
        for subdir in view_dir.iterdir():
            if subdir.is_dir() and subdir.name.lower().startswith("bad"):
                bad_cat = subdir.name
                val1_dir = subdir / "val1"
                if not val1_dir.exists(): continue
                
                anno_key = f"{view_key}_{bad_cat}" 
                xml_path = ANNOTATIONS_DIR / f"{anno_key}.xml"
                gt_boxes_map = load_xml_boxes(xml_path)
                
                images = list(val1_dir.glob("*.jpg")) + list(val1_dir.glob("*.png"))
                for img_path in tqdm(images, desc=f"Eval {bad_cat}"):
                    gt_boxes = gt_boxes_map.get(img_path.name, [])
                    is_faulty, vis_data = evaluate_image(model, img_path, gt_boxes)
                    
                    if is_faulty:
                        TP += 1
                    else:
                        FN += 1
                        out_name = f"FN_{view_key}_{bad_cat}_{img_path.name}"
                        save_visualization(vis_data, MISMATCH_DIR / out_name)

        # Progress Update
        m = compute_metrics(TP, FP, FN, TN)
        print(f"Current Metrics: Acc={m['Accuracy']:.3f}, F1={m['F1']:.3f}, MCC={m['MCC']:.3f}, Spec={m['Specificity']:.3f}")

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    m = compute_metrics(TP, FP, FN, TN)
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
    
    print(f"\nTotal Images: {TP+FP+FN+TN}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Mismatches saved to: {MISMATCH_DIR}")
    
    # Save Results to File (Append Mode)
    with open(RESULTS_FILE, "a") as f:
        f.write("\n\n" + "-"*50 + "\n")
        f.write(f"RUN: {RUN_NAME}\n")
        f.write("-"*50 + "\n")
        for k, v in m.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"Total: {TP+FP+FN+TN} (TP={TP}, FP={FP}, FN={FN}, TN={TN})\n")
        f.write(f"Saved Mismatches: {MISMATCH_DIR}\n")
        
    # Re-read file to generate full comparison including just-added run
    all_runs = extract_all_results(RESULTS_FILE)
    if all_runs:
        # Write Comparison Table
        with open(RESULTS_FILE, "a") as f:
            f.write("\n" + "="*80 + "\n")
            f.write("COMPARISON: All Runs\n")
            f.write("="*80 + "\n")
            
            # Header: Metric | Run 1 | Run 2 | ...
            run_names = list(all_runs.keys())
            # Ensure current run is last? Dictionary order depends on insertion (Python 3.7+ is insertion ordered).
            # The file reading order should preserve run order.
            
            header = f"{'Metric':<15}"
            for name in run_names:
                # Truncate name if too long
                short_name = (name[:18] + '..') if len(name) > 20 else name
                header += f" {short_name:<20}"
            f.write(header + "\n")
            f.write("-" * (15 + 21*len(run_names)) + "\n")
            
            for k in ["Accuracy", "F1", "Specificity", "Recall", "Precision", "MCC"]:
                row = f"{k:<15}"
                for name in run_names:
                    val = all_runs[name].get(k, 0.0)
                    row += f" {val:<20.4f}"
                f.write(row + "\n")
            f.write("="*80 + "\n")
            
    print(f"Results appended to: {RESULTS_FILE.absolute()}")

if __name__ == "__main__":
    main()
