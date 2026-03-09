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
import xml.etree.ElementTree as ET
import time

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
DATASET_ROOT = Path(r"c:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\dataset without bg")
RESULTS_FILE = Path("comprehensive_evaluation_results.txt")
MODELS_DIR = Path(r"c:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\reduced_models")
ANNOTATIONS_DIR = Path(r"c:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\annotations_new_dataset")

IMG_SIZE = 1024
CROP_PX = 180
ANOMALY_THRESHOLD = 4.25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

SIDES = ["front", "back", "side1", "side2"]
ANGLES = ["0", "45_unflipped", "45 - flipped"]

# --------------------------------------------------------------
# UTILS
# --------------------------------------------------------------
def make_foreground_mask(pil_rgb: Image.Image) -> np.ndarray:
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
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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
            x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
            
            feats = []
            for feat in [x2, x3]:
                b, c, h, w = feat.shape
                feat = feat.view(b, c, h*w).permute(0, 2, 1)
                feats.append(feat)
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
                vals, _ = d.topk(k=9, dim=1, largest=False)
                mins.append(vals.mean(dim=1))
                
        return torch.cat(mins, dim=0)

# --------------------------------------------------------------
# CORE PIPELINE LOGIC
# --------------------------------------------------------------
def train_model(train_dirs, model_path, run_desc):
    if model_path.exists():
        print(f"[{run_desc}] Model exists: {model_path.name}")
        return
        
    print(f"[{run_desc}] Training model from {len(train_dirs)} directories...")
    model = PatchCore().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    images = []
    for d in train_dirs:
        if d.exists():
            images.extend(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg")))
        
    if not images:
        print(f"Skipping training: No images found.")
        return
        
    all_feats = []
    batch_size = 4
    
    for i in tqdm(range(0, len(images), batch_size), desc=f"[{run_desc}] Extracting"):
        batch_files = images[i:i+batch_size]
        batch_imgs = []
        for f in batch_files:
            pil_img = Image.open(f).convert("RGB")
            
            # MASK BORDER
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
    print(f"[{run_desc}] Saved model to {model_path.name}")

def evaluate_image(model, img_path, gt_boxes=None):
    pil_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    
    if CROP_PX > 0:
        img_np = np.array(pil_img)
        crop_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        x1, y1 = CROP_PX, CROP_PX
        x2, y2 = orig_w - CROP_PX, orig_h - CROP_PX
        
        if x2 > x1 and y2 > y1:
            crop_mask[y1:y2, x1:x2] = 1
            img_np[crop_mask == 0] = 0
            pil_img = Image.fromarray(img_np)

    mask = make_foreground_mask(pil_img)
    mask_bool = (mask > 0).astype(np.uint8)
    
    img_np = np.array(pil_img)
    img_with_mask = img_np.copy()
    img_with_mask[mask == 0] = 0 
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor = transform(Image.fromarray(img_with_mask)).unsqueeze(0).to(device)
    
    feats = model.extract_features(tensor)
    scores = model.compute_anomaly_score_map(feats, device).cpu().numpy()
    
    N = len(scores)
    side_len = int(np.sqrt(N))
    score_map = scores.reshape(side_len, side_len)
    score_map = cv2.resize(score_map, (orig_w, orig_h))
    
    masked_score = score_map.copy()
    masked_score[mask_bool == 0] = 0 
    max_score = masked_score.max()
    return max_score > ANOMALY_THRESHOLD

def evaluate_dataset(model, angle, side, desc):
    """Evaluates the model on all test sets for a specific angle and side"""
    TP = FP = FN = TN = 0
    
    angle_dir = angle
    if angle == "45":
        angle_dir = "45_unflipped" # Default if passed improperly, handled in wrapper 
        
    view_dir = DATASET_ROOT / (angle if "45" not in angle else "45") / angle / side
    if not view_dir.exists():
        return 0, 0, 0, 0
        
    # Good tests (testgood1, testgood2)
    for test_dir_name in ["testgood1", "testgood2"]:
        test_dir = view_dir / test_dir_name
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            for img_path in tqdm(images, desc=f"[{desc}] Eval {angle} {test_dir_name}", leave=False):
                is_faulty = evaluate_image(model, img_path)
                if not is_faulty: TN += 1
                else: FP += 1
                
    # Bad tests (bad1, bad2, bad3 -> val1)
    for bad_cat in ["bad1", "bad2", "bad3"]:
        val1_dir = view_dir / bad_cat / "val1"
        if val1_dir.exists():
            images = list(val1_dir.glob("*.jpg")) + list(val1_dir.glob("*.png"))
            for img_path in tqdm(images, desc=f"[{desc}] Eval {angle} {bad_cat}", leave=False):
                is_faulty = evaluate_image(model, img_path)
                if is_faulty: TP += 1
                else: FN += 1
                
    return TP, FP, FN, TN

def load_model(path):
    model = PatchCore().to(device)
    model.memory_bank = torch.load(path, map_location=device)
    return model

# --------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------
def main():
    print("Starting Comprehensive Model Evaluation...")
    overall_start = time.time()
    
    results_map = {
        "Config A (Individual)": {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "time_train": 0, "time_eval": 0},
        "Config B (2-Angle Combined + Fallback)": {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "time_train": 0, "time_eval": 0},
        "Config C (3-Angle Combined)": {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "time_train": 0, "time_eval": 0}
    }
    
    for side in SIDES:
        print(f"\n{'='*60}")
        print(f"Processing Side: {side}")
        print(f"{'='*60}")
        
        # ----------------------------------------------------------------------
        # Data Declarations
        # ----------------------------------------------------------------------
        dir_0_train_g1 = DATASET_ROOT / "0" / side / "good1"
        dir_45un_train_g1 = DATASET_ROOT / "45" / "45_unflipped" / side / "good1"
        dir_45fl_train_g1 = DATASET_ROOT / "45" / "45 - flipped" / side / "good1"
        
        # ----------------------------------------------------------------------
        # 1. Config A: Individual Models
        # ----------------------------------------------------------------------
        print("\n--- PHASE 1: CONFIG A (Individual Models) ---")
        train_A_start = time.time()
        
        mod_0_path = MODELS_DIR / f"individual_0_{side}.pth"
        mod_45un_path = MODELS_DIR / f"individual_45unflipped_{side}.pth"
        mod_45fl_path = MODELS_DIR / f"individual_45flipped_{side}.pth"
        
        train_model([dir_0_train_g1], mod_0_path, "ConfigA-0")
        train_model([dir_45un_train_g1], mod_45un_path, "ConfigA-45un")
        train_model([dir_45fl_train_g1], mod_45fl_path, "ConfigA-45fl")
        
        results_map["Config A (Individual)"]["time_train"] += (time.time() - train_A_start)
        
        # EVAL A
        eval_A_start = time.time()
        
        # Execute 0 degree
        m0 = load_model(mod_0_path)
        tp, fp, fn, tn = evaluate_dataset(m0, "0", side, "ConfA_0")
        results_map["Config A (Individual)"]["TP"] += tp
        results_map["Config A (Individual)"]["FP"] += fp
        results_map["Config A (Individual)"]["FN"] += fn
        results_map["Config A (Individual)"]["TN"] += tn
        
        # Execute 45 unflipped
        m45un = load_model(mod_45un_path)
        tp, fp, fn, tn = evaluate_dataset(m45un, "45_unflipped", side, "ConfA_45un")
        results_map["Config A (Individual)"]["TP"] += tp
        results_map["Config A (Individual)"]["FP"] += fp
        results_map["Config A (Individual)"]["FN"] += fn
        results_map["Config A (Individual)"]["TN"] += tn
        
        # Execute 45 flipped
        m45fl = load_model(mod_45fl_path)
        tp, fp, fn, tn = evaluate_dataset(m45fl, "45 - flipped", side, "ConfA_45fl")
        results_map["Config A (Individual)"]["TP"] += tp
        results_map["Config A (Individual)"]["FP"] += fp
        results_map["Config A (Individual)"]["FN"] += fn
        results_map["Config A (Individual)"]["TN"] += tn
        
        results_map["Config A (Individual)"]["time_eval"] += (time.time() - eval_A_start)
        
        # Free memory
        del m0, m45un, m45fl
        torch.cuda.empty_cache()


        # ----------------------------------------------------------------------
        # 2. Config B: 2-Angle Combined (0 + 45unflipped)
        # ----------------------------------------------------------------------
        print("\n--- PHASE 2: CONFIG B (2-Angle Combined + Fallback) ---")
        train_B_start = time.time()
        
        mod_B_path = MODELS_DIR / f"combined_2angle_{side}.pth"
        train_model([dir_0_train_g1, dir_45un_train_g1], mod_B_path, "ConfigB-Comb")
        
        results_map["Config B (2-Angle Combined + Fallback)"]["time_train"] += (time.time() - train_B_start)
        
        # EVAL B
        eval_B_start = time.time()
        m_comb2 = load_model(mod_B_path)
        
        # Execute 0 degree (using combined)
        tp, fp, fn, tn = evaluate_dataset(m_comb2, "0", side, "ConfB_0")
        results_map["Config B (2-Angle Combined + Fallback)"]["TP"] += tp
        results_map["Config B (2-Angle Combined + Fallback)"]["FP"] += fp
        results_map["Config B (2-Angle Combined + Fallback)"]["FN"] += fn
        results_map["Config B (2-Angle Combined + Fallback)"]["TN"] += tn
        
        # Execute 45 unflipped (using combined)
        tp, fp, fn, tn = evaluate_dataset(m_comb2, "45_unflipped", side, "ConfB_45un")
        results_map["Config B (2-Angle Combined + Fallback)"]["TP"] += tp
        results_map["Config B (2-Angle Combined + Fallback)"]["FP"] += fp
        results_map["Config B (2-Angle Combined + Fallback)"]["FN"] += fn
        results_map["Config B (2-Angle Combined + Fallback)"]["TN"] += tn
        
        # Execute 45 flipped (using FALLBACK INDIVIDUAL)
        print("[ConfigB] Falling back to Individual 45-flipped model for 45-flipped test set")
        m45fl_fallback = load_model(mod_45fl_path)
        tp, fp, fn, tn = evaluate_dataset(m45fl_fallback, "45 - flipped", side, "ConfB_FB_45fl")
        results_map["Config B (2-Angle Combined + Fallback)"]["TP"] += tp
        results_map["Config B (2-Angle Combined + Fallback)"]["FP"] += fp
        results_map["Config B (2-Angle Combined + Fallback)"]["FN"] += fn
        results_map["Config B (2-Angle Combined + Fallback)"]["TN"] += tn
        
        results_map["Config B (2-Angle Combined + Fallback)"]["time_eval"] += (time.time() - eval_B_start)
        
        del m_comb2, m45fl_fallback
        torch.cuda.empty_cache()


        # ----------------------------------------------------------------------
        # 3. Config C: 3-Angle Combined (0 + 45unflipped + 45flipped)
        # ----------------------------------------------------------------------
        print("\n--- PHASE 3: CONFIG C (3-Angle Combined) ---")
        train_C_start = time.time()
        
        mod_C_path = MODELS_DIR / f"combined_3angle_{side}.pth"
        train_model([dir_0_train_g1, dir_45un_train_g1, dir_45fl_train_g1], mod_C_path, "ConfigC-Comb")
        
        results_map["Config C (3-Angle Combined)"]["time_train"] += (time.time() - train_C_start)
        
        # EVAL C
        eval_C_start = time.time()
        m_comb3 = load_model(mod_C_path)
        
        # Execute all three against combined
        tp, fp, fn, tn = evaluate_dataset(m_comb3, "0", side, "ConfC_0")
        results_map["Config C (3-Angle Combined)"]["TP"] += tp
        results_map["Config C (3-Angle Combined)"]["FP"] += fp
        results_map["Config C (3-Angle Combined)"]["FN"] += fn
        results_map["Config C (3-Angle Combined)"]["TN"] += tn

        tp, fp, fn, tn = evaluate_dataset(m_comb3, "45_unflipped", side, "ConfC_45un")
        results_map["Config C (3-Angle Combined)"]["TP"] += tp
        results_map["Config C (3-Angle Combined)"]["FP"] += fp
        results_map["Config C (3-Angle Combined)"]["FN"] += fn
        results_map["Config C (3-Angle Combined)"]["TN"] += tn

        tp, fp, fn, tn = evaluate_dataset(m_comb3, "45 - flipped", side, "ConfC_45fl")
        results_map["Config C (3-Angle Combined)"]["TP"] += tp
        results_map["Config C (3-Angle Combined)"]["FP"] += fp
        results_map["Config C (3-Angle Combined)"]["FN"] += fn
        results_map["Config C (3-Angle Combined)"]["TN"] += tn

        results_map["Config C (3-Angle Combined)"]["time_eval"] += (time.time() - eval_C_start)

        del m_comb3
        torch.cuda.empty_cache()

    overall_time = time.time() - overall_start

    # ----------------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------------
    print("\n\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    # Calculate derived metrics
    metrics_map = {}
    for cfg, counts in results_map.items():
        m = compute_metrics(counts["TP"], counts["FP"], counts["FN"], counts["TN"])
        m["Total"] = counts["TP"] + counts["FP"] + counts["FN"] + counts["TN"]
        m["TP"], m["FP"], m["FN"], m["TN"] = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
        metrics_map[cfg] = m
    
    # Print Table
    metric_keys = ["Accuracy", "F1", "Specificity", "Recall", "Precision", "MCC"]
    
    header = f"{'Metric':<15}"
    for cfg in results_map.keys():
        short_name = cfg[:15] + ".." if len(cfg) > 17 else cfg
        header += f" {short_name:<18}"
    print(header)
    print("-" * 75)
    
    for k in metric_keys:
        row = f"{k:<15}"
        for cfg in results_map.keys():
            val = metrics_map[cfg][k]
            row += f" {val:<18.4f}"
        print(row)
        
    print("-" * 75)
    row_tot = f"{'Evaluated':<15}"
    for cfg in results_map.keys():
        row_tot += f" {metrics_map[cfg]['Total']:<18}"
    print(row_tot)
    
    print("="*80)
    print("\nTiming Log")
    for cfg in results_map.keys():
        c = results_map[cfg]
        print(f"{cfg}: Train={c['time_train']:.2f}s, Eval={c['time_eval']:.2f}s")
    print(f"Overall Script Time: {overall_time:.2f}s")
    
    # Write to File
    with open(RESULTS_FILE, "w") as f:
        f.write("COMPREHENSIVE EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(header + "\n")
        f.write("-" * 75 + "\n")
        for k in metric_keys:
            row = f"{k:<15}"
            for cfg in results_map.keys():
                val = metrics_map[cfg][k]
                row += f" {val:<18.4f}"
            f.write(row + "\n")
        f.write("-" * 75 + "\n")
        
        row_tot = f"{'Evaluated':<15}"
        for cfg in results_map.keys():
            row_tot += f" {metrics_map[cfg]['Total']:<18}"
        f.write(row_tot + "\n")
        
        f.write("="*80 + "\n\n")
        f.write("RAW COUNTS\n")
        f.write("-" * 40 + "\n")
        for cfg in results_map.keys():
            m = metrics_map[cfg]
            f.write(f"{cfg}:\n")
            f.write(f"  TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, TN={m['TN']}\n")
            
        f.write("\nTIMING LOG\n")
        f.write("-" * 40 + "\n")
        for cfg in results_map.keys():
            c = results_map[cfg]
            f.write(f"{cfg}: Train={c['time_train']:.2f}s, Eval={c['time_eval']:.2f}s\n")
        f.write(f"Overall Script Time: {overall_time:.2f}s\n")
        
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
