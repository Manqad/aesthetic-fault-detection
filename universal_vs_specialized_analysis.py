
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import wide_resnet101_2, Wide_ResNet101_2_Weights
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
import json
import matplotlib.pyplot as plt
import sys

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\acpart2")
MODELS_DIR = Path(r"C:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\models\comparison")
UNIVERSAL_MODEL_PATH = Path(r"C:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\models\patchcore_universal_good.pth")
RESULTS_DIR = Path(r"C:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\universal_results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ANGLES = ["0", "45upright"]
SIDES = ["front", "back", "side1", "side2"]
# Note: "badgood" is a bad category in the user's dataset
BAD_CATS = ["bad1", "bad2", "bad3", "badgood"] 

IMG_SIZE = 1024
ANOMALY_THRESHOLD = 4.25
MAX_INFERENCE_BANK_SIZE = 200000 

# --------------------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------------------
class PatchCoreInference(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = wide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1)
        self.layer2 = nn.Sequential(*list(self.backbone.children())[:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[:7])
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.memory_bank = None

    def extract_features(self, x):
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        feats = []
        for feat in [x2, x3]:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h*w).permute(0, 2, 1)
            feats.append(feat)
        return torch.cat(feats, dim=2)

    def compute_anomaly_score(self, features, device, chunk=1024):
        if self.memory_bank is None:
            raise ValueError("Memory bank not loaded.")
        
        bank = self.memory_bank
        if bank.dim() == 3:
            bank = bank.view(-1, bank.size(-1))
            
        # Subsample bank if too large (for speed on CPU)
        if bank.shape[0] > MAX_INFERENCE_BANK_SIZE:
             step = bank.shape[0] // MAX_INFERENCE_BANK_SIZE
             bank = bank[::step][:MAX_INFERENCE_BANK_SIZE]

        bank = bank.to(device)
        feat = features.view(-1, features.size(-1)).to(device)
        
        if device.type == "cpu":
             bank = bank.float()
             feat = feat.float()
        elif bank.dtype == torch.float16:
             feat = feat.half()
        else:
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
    
    def forward(self, x):
        return self.extract_features(x)

# Optional Background Removal (Consistent with training)
try:
    from rembg import remove as rembg_remove
    _HAS_REMBG = True
except ImportError:
    _HAS_REMBG = False

def make_foreground_mask(pil_rgb: Image.Image) -> np.ndarray:
    if not _HAS_REMBG:
        w, h = pil_rgb.size
        return np.ones((h, w), dtype=np.uint8) * 255
    rgba = pil_rgb.convert("RGBA")
    rgba_removed = rembg_remove(rgba)
    alpha = np.array(rgba_removed)[:, :, 3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.erode(alpha, k, iterations=1)
    alpha = cv2.dilate(alpha, k, iterations=1)
    return alpha

# --------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------
def detect_anomaly(img_path, model, device):
    try:
        pil_rgb = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return False, 0.0

    orig_w, orig_h = pil_rgb.size
    
    # Generate Mask
    mask = make_foreground_mask(pil_rgb)
    mask_bool = (mask > 0).astype(np.uint8)
    
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = tf(pil_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(tensor)
        anomaly_scores = model.compute_anomaly_score(features, device).numpy()
    
    N = len(anomaly_scores)
    side = int(np.sqrt(N))
    score_map = anomaly_scores.reshape(side, side)
    score_map = cv2.resize(score_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # Apply Mask
    masked_score = score_map.copy()
    masked_score[mask_bool == 0] = 0.0
    
    # Binary decision
    binary = (masked_score > ANOMALY_THRESHOLD).astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    is_faulty = False
    for lbl in range(1, num_labels):
        area = stats[lbl, 4]
        if area >= 5: # Min area filter
             is_faulty = True
             break
             
    # Return max score of the valid (masked) area
    return is_faulty, masked_score.max()

def get_test_images():
    """
    Returns a list of test items.
    Each item: {
        'path': Path,
        'label': 'good' | 'bad',
        'angle': str,
        'side': str
    }
    """
    test_set = []
    
    print("Collecting test images...")
    for ang in ANGLES:
        for side in SIDES:
            # 1. Collect BAD images (Positive samples: bad1, bad2, bad3)
            # EXCLUDING 'badgood' from this loop to treat it as GOOD
            actual_bad_cats = [c for c in BAD_CATS if c != "badgood"]
            for cat in actual_bad_cats:
                bad_dir = BASE_DIR / ang / side / "bad" / cat
                if bad_dir.exists():
                    imgs = list(bad_dir.glob("*.jpg")) + list(bad_dir.glob("*.png"))
                    for i in imgs:
                        test_set.append({'path': i, 'label': 'bad', 'angle': ang, 'side': side})
            
            # 2. Collect GOOD test images (Negative samples)
            # User specified: Use 'badgood' folder images as the ONLY valid test good images
            # (ignoring the main 'good' folder which was used for training)
            badgood_dir = BASE_DIR / ang / side / "bad" / "badgood"
            if badgood_dir.exists():
                imgs = list(badgood_dir.glob("*.jpg")) + list(badgood_dir.glob("*.png"))
                for i in imgs:
                    test_set.append({'path': i, 'label': 'good', 'angle': ang, 'side': side})
                    
    print(f"Total Test Images: {len(test_set)}")
    bad_count = sum(1 for x in test_set if x['label']=='bad')
    good_count = sum(1 for x in test_set if x['label']=='good')
    print(f"  - Bad (Faulty): {bad_count}")
    print(f"  - Good (Clean): {good_count} ( sourced from 'badgood' folders )")
    
    # Validation
    if bad_count == 0 or good_count == 0:
        print("WARNING: Dataset imbalance found. Ensure 'badgood' folders exist and contain images.")
        
    return test_set

def calculate_metrics(results):
    TP = results['TP']
    FP = results['FP']
    TN = results['TN']
    FN = results['FN']
    
    total = TP + FP + TN + FN
    if total == 0: return {}
    
    acc = (TP + TN) / total
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    mdr = FN / (FN + TP) if (FN + TP) > 0 else 0
    
    denom = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    mcc = (TP * TN - FP * FN) / denom if denom > 0 else 0
    
    return {
        "Accuracy": acc,
        "MCC": mcc,
        "F1-Score": f1,
        "Precision": prec,
        "Recall": rec,
        "FAR": far,
        "MDR": mdr,
        "Confusion Matrix": {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
    }

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    test_set = get_test_images()
    if not test_set:
        print("No images found!")
        return

    # Initialize Model Wrapper
    model = PatchCoreInference().to(device)

    # ==========================================================
    # 1. EVALUATE SPECIALIZED MODELS (SKIPPED - USING CACHED RESULTS)
    # ==========================================================
    print("\n" + "="*50)
    print(" EVALUATING SPECIALIZED MODELS (SKIPPED)")
    print(" Using cached results from previous run for comparison.")
    print("="*50)
    
    # User provided results from previous run (step 144)
    # Accuracy        | 0.9310
    # F1-Score        | 0.9581
    # FAR             | 0.2250
    # MDR             | 0.0338
    # MCC             | 0.7642
    
    # We populate the metrics directly since we don't have the raw counts anymore
    metrics_spec = {
        "Accuracy": 0.9310,
        "F1-Score": 0.9581,
        "FAR": 0.2250,
        "MDR": 0.0338,
        "MCC": 0.7642,
        "Precision": 0.0, # Placeholder
        "Recall": 0.9662  # Approx deriv from MDR
    }

    # ==========================================================
    # 2. EVALUATE UNIVERSAL MODEL
    # ==========================================================
    print("\n" + "="*50)
    print(" EVALUATING UNIVERSAL MODEL")
    print("="*50)
    
    res_universal = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    
    # Load Universal Bank
    if not UNIVERSAL_MODEL_PATH.exists():
        print(f"Critical: Universal model not found at {UNIVERSAL_MODEL_PATH}")
    else:
        try:
            print(f"Loading Universal Model: {UNIVERSAL_MODEL_PATH}")
            state_dict = torch.load(UNIVERSAL_MODEL_PATH, map_location=device, weights_only=True)
            model.memory_bank = state_dict
            
            for item in tqdm(test_set, desc="Universal Eval"):
                is_faulty, _ = detect_anomaly(item['path'], model, device)
                is_bad_ground_truth = (item['label'] == 'bad')
                
                if is_bad_ground_truth:
                    if is_faulty: res_universal['TP'] += 1
                    else: res_universal['FN'] += 1
                else:
                    if is_faulty: res_universal['FP'] += 1
                    else: res_universal['TN'] += 1
                    
        except Exception as e:
            print(f"Error evaluating universal model: {e}")

    # ==========================================================
    # 3. REPORTING
    # ==========================================================
    # metrics_spec is already set above
    metrics_univ = calculate_metrics(res_universal)

    # Print Report
    print("\n" + "="*50)
    print(" FINAL COMPARISON REPORT")
    print("="*50)
    print(f"{'Metric':<15} | {'Specialized':<15} | {'Universal':<15}")
    print("-" * 50)
    for m in ["Accuracy", "F1-Score", "FAR", "MDR", "MCC"]:
        v_s = metrics_spec.get(m, 0.0)
        v_u = metrics_univ.get(m, 0.0)
        print(f"{m:<15} | {v_s:.4f}          | {v_u:.4f}")
    
    # Save JSON
    final_output = {
        "Specialized": metrics_spec,
        "Universal": metrics_univ
    }
    
    out_file = RESULTS_DIR / "universal_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=4)
        print(f"\nResults saved to {out_file}")

    # Generate Plots
    metrics_list = ["Accuracy", "MCC", "F1-Score", "FAR", "MDR"]
    x = np.arange(len(metrics_list))
    width = 0.35
    
    vals_s = [metrics_spec.get(m, 0.0) for m in metrics_list]
    vals_u = [metrics_univ.get(m, 0.0) for m in metrics_list]
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, vals_s, width, label='Specialized')
    plt.bar(x + width/2, vals_u, width, label='Universal')
    
    plt.ylabel('Score')
    plt.title('Specialized vs Universal Model Performance')
    plt.xticks(x, metrics_list)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "universal_comparison_chart.png")
    print(f"Chart saved to {RESULTS_DIR / 'universal_comparison_chart.png'}")

if __name__ == "__main__":
    main()
