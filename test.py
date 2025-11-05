# test.py — Raw-distance thresholding + FG mask + NMS + visualization
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET

# Optional background remover
try:
    from rembg import remove as rembg_remove
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
IMG_SIZE = 512

# ---- RAW THRESHOLD on distance map (no z-score / hysteresis) ----
ANOMALY_THRESHOLD = 4.3  # tune higher to reduce FPs, lower to catch more

# Component post-processing
MIN_BOX_W = 6
MIN_BOX_H = 6
MIN_CONTOUR_AREA = 5
MAX_ASPECT_RATIO = 10.0

# Non-maximum suppression
NMS_IOU = 0.30

# Foreground masking
BACKGROUND_REMOVAL = True       # uses rembg alpha; if rembg missing -> full FG
BG_ERODE = 1
BG_DILATE = 1

COMPARISON_DIR = Path("d:/fyp/aesthetic-fault-detection/comparisons")
COMPARISON_DIR.mkdir(exist_ok=True)
SAVED_CATEGORIES = {"bad1": False, "bad2": False, "bad3": False}

# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------
def nms_xyxy(boxes, scores, iou_thr=0.3):
    """Non-maximum suppression for (x1,y1,x2,y2)."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def box_mean_score(score_map, x1, y1, x2, y2):
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(score_map.shape[1]-1, int(x2)); y2 = min(score_map.shape[0]-1, int(y2))
    crop = score_map[y1:y2+1, x1:x2+1]
    if crop.size == 0:
        return 0.0
    return float(crop.mean())

def load_xml_boxes(xml_path: Path):
    """Return dict: name -> list of [x,y,w,h]."""
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

def make_foreground_mask(pil_rgb: Image.Image) -> np.ndarray:
    """
    Returns uint8 mask in original image size: 255 for foreground, 0 for background.
    Uses rembg if available; otherwise returns full-ones mask.
    """
    if not BACKGROUND_REMOVAL or not _HAS_REMBG:
        if BACKGROUND_REMOVAL and not _HAS_REMBG:
            print("[WARN] BACKGROUND_REMOVAL=True but rembg not available. Proceeding without masking.")
        w, h = pil_rgb.size
        return np.ones((h, w), dtype=np.uint8) * 255

    rgba = pil_rgb.convert("RGBA")
    rgba_removed = rembg_remove(rgba)  # RGBA with alpha
    alpha = np.array(rgba_removed)[:, :, 3]  # 0..255
    # Simple cleanup
    if BG_ERODE > 0 or BG_DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(BG_ERODE):
            alpha = cv2.erode(alpha, k, iterations=1)
        for _ in range(BG_DILATE):
            alpha = cv2.dilate(alpha, k, iterations=1)
    return alpha

# --------------------------------------------------------------
# PatchCore Inference
# --------------------------------------------------------------
class PatchCoreInference(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.layer2 = nn.Sequential(*list(self.backbone.children())[:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[:7])
        self.memory_bank = None  # [M,1536] or [N,4096,1536]

    def extract_features(self, x):
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        feats = []
        for feat in [x2, x3]:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h*w).permute(0, 2, 1)  # [B, HW, C]
            feats.append(feat)
        return torch.cat(feats, dim=2)  # [B, 4096, 1536]

    def compute_anomaly_score(self, features, device, chunk=8192):
        if self.memory_bank is None:
            raise ValueError("Memory bank not loaded.")
        bank = self.memory_bank
        if bank.dim() == 3:
            bank = bank.view(-1, bank.size(-1))
        bank = bank.float().cpu()
        feat = features.view(-1, features.size(-1)).float().cpu()  # [4096,1536]
        mins = []
        for i in range(0, feat.size(0), chunk):
            f = feat[i:i+chunk]
            d = torch.cdist(f, bank, p=2)
            mins.append(d.min(dim=1).values)
        return torch.cat(mins, dim=0)  # [4096] CPU

    def forward(self, x):
        return self.extract_features(x)

# --------------------------------------------------------------
# SAVE COMPARISON (ONE PER CATEGORY)
# --------------------------------------------------------------
def save_comparison(img_pred, img_gt, img_name, cat):
    comparison = np.hstack([img_pred, img_gt])
    save_path = COMPARISON_DIR / f"comparison_{cat}_{Path(img_name).stem}.jpg"
    cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"  SAVED: {save_path}")

# --------------------------------------------------------------
# DETECT ANOMALIES (raw-distance thresholding)
# --------------------------------------------------------------
def detect_anomalies(img_path: str, memory_bank_path: str, gt_boxes: list, cat: str):
    """
    Returns:
        img_pred: RGB np.array with predicted boxes (green)
        pred_boxes_eval: list of [x,y,w,h]
        score_map: raw distance map resized to original image size
    """
    global SAVED_CATEGORIES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCoreInference().to(device)
    model.eval()

    # Load memory bank (supports both [N,4096,1536] and [M,1536])
    model.memory_bank = torch.load(memory_bank_path, map_location="cpu")
    print(f"[DEBUG] bank shape before view: {tuple(model.memory_bank.shape)}")
    if model.memory_bank.dim() == 3:
        model.memory_bank = model.memory_bank.view(-1, model.memory_bank.size(-1))
    print(f"[DEBUG] bank shape after  view: {tuple(model.memory_bank.shape)}")

    # Load image + foreground mask
    pil_rgb = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_rgb.size
    fg_mask = make_foreground_mask(pil_rgb)  # 0..255
    fg_mask_bool = (fg_mask > 0).astype(np.uint8)

    # Preprocess for backbone
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = tf(pil_rgb).unsqueeze(0).to(device)

    # Extract + score
    with torch.no_grad():
        features = model(tensor)                     # [1,4096,1536]
        anomaly_scores = model.compute_anomaly_score(features, device).numpy()

    # Reshape + resize to original resolution  (this is the RAW distance map)
    score_map = anomaly_scores.reshape(64, 64)
    score_map = cv2.resize(score_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # ---------- RAW THRESHOLDING (no z-normalization) ----------
    # Apply foreground mask: suppress background
    masked_score = score_map.copy()
    masked_score[fg_mask_bool == 0] = 0.0

    # Binary map by fixed threshold on raw distances
    binary = (masked_score > ANOMALY_THRESHOLD).astype(np.uint8) * 255

    # Morphology (same as before, just to tidy blobs)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5)

    # Connected components on binary
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    kept_xyxy, kept_scores = [], []

    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl, 0], stats[lbl, 1], stats[lbl, 2], stats[lbl, 3], stats[lbl, 4]
        if w < MIN_BOX_W or h < MIN_BOX_H:
            continue
        if area < MIN_CONTOUR_AREA:
            continue
        aspect = max(w / h, h / w) if min(w, h) > 0 else float("inf")
        if aspect > MAX_ASPECT_RATIO:
            continue

        x1_t, y1_t, x2_t, y2_t = x, y, x + w - 1, y + h - 1

        # Clamp
        x1_t = max(0, x1_t); y1_t = max(0, y1_t)
        x2_t = min(x2_t, orig_w - 1); y2_t = min(y2_t, orig_h - 1)

        # Optional shave to reduce border noise
        if (x2_t - x1_t) > 6 and (y2_t - y1_t) > 6:
            x1_t += 1; y1_t += 1; x2_t -= 1; y2_t -= 1

        # Score = mean RAW distance inside the box
        m = box_mean_score(score_map, x1_t, y1_t, x2_t, y2_t)
        kept_xyxy.append([x1_t, y1_t, x2_t, y2_t])
        kept_scores.append(m)

    # NMS over boxes
    keep_idx = nms_xyxy(kept_xyxy, kept_scores, iou_thr=NMS_IOU)
    pred_boxes_xyxy = [kept_xyxy[i] for i in keep_idx]

    # Convert to (x,y,w,h) for evaluation
    pred_boxes_eval = [[x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)] for (x1, y1, x2, y2) in pred_boxes_xyxy]

    # Draw
    img_pred = np.array(pil_rgb)
    for (x1, y1, x2, y2) in pred_boxes_xyxy:
        cv2.rectangle(img_pred, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    img_gt = np.array(pil_rgb)
    for (x, y, w, h) in gt_boxes:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 0, 255), 3)

    if not SAVED_CATEGORIES.get(cat, False):
        save_comparison(img_pred, img_gt, img_path, cat)
        SAVED_CATEGORIES[cat] = True

    nz_bin = int((binary > 0).sum())
    print(f"[DEBUG] preds(after NMS): {len(pred_boxes_eval)}  gt: {len(gt_boxes)}  "
          f"thr: {ANOMALY_THRESHOLD:.3f}  nz_bin: {nz_bin}  rembg:{_HAS_REMBG}")

    return img_pred, pred_boxes_eval, score_map

# --------------------------------------------------------------
# EVALUATION (IoU-based — unchanged)
# --------------------------------------------------------------
def iou(a, b):
    xa1, ya1, wa, ha = a
    xb1, yb1, wb, hb = b
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb2, yb2 = xb1 + wb, yb1 + hb
    ix1 = max(xa1, xb1); iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2); iy2 = min(ya2, yb2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = wa * ha + wb * hb - inter
    return inter / union if union > 0 else 0.0

def evaluate_detection(pred, gt, iou_thr=0.5, debug=False, img_name=None):
    """
    pred, gt: lists of [x, y, w, h]
    iou_thr: IoU threshold for a match (default 0.5)
    debug: if True, prints best IoU per prediction
    """
    if not pred and not gt: return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
    if not pred: return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    if not gt:   return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    if debug:
        print(f"[IOU-DBG] {img_name or ''} preds={len(pred)} gts={len(gt)} thr={iou_thr}")
        for pi, p in enumerate(pred):
            best, best_i = 0.0, -1
            for gi, g in enumerate(gt):
                v = iou(p, g)
                if v > best: best, best_i = v, gi
            print(f"  pred[{pi}]={p}  best_iou={best:.3f} -> gt[{best_i}]={gt[best_i] if best_i>=0 else None}")

    tp = 0
    matched = [False] * len(gt)
    for p in pred:
        for i, g in enumerate(gt):
            if not matched[i] and iou(p, g) >= iou_thr:
                tp += 1
                matched[i] = True
                break
    fp = len(pred) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1_score": f1}

# --------------------------------------------------------------
# POST-TEST VISUALIZATION (one per category)
# --------------------------------------------------------------
def visualize_after_testing(angle: str,
                            side: str,
                            base_dir: str | Path,
                            annotations_dir: str | Path,
                            memory_bank_dir: str | Path,
                            categories=None):
    """
    After your testing loop finishes, call this once to save one comparison
    (pred vs GT) per category into COMPARISON_DIR.
    """
    global SAVED_CATEGORIES

    angle_tag = str(angle).replace("º", "").replace("°", "")
    base_dir = Path(base_dir)
    annotations_dir = Path(annotations_dir)
    memory_bank_dir = Path(memory_bank_dir)

    mem_path = memory_bank_dir / f"patchcore_{angle}_{side}.pth"
    if not mem_path.exists():
        print(f"[SKIP] Memory bank not found: {mem_path}")
        return

    if categories is None:
        categories = ["bad1", "bad2", "bad3"]

    # ensure we save exactly one image per category in this call
    SAVED_CATEGORIES = {c: False for c in categories}

    for cat in categories:
        xml_file = f"{angle_tag}-{side}-bad-{cat}.xml"
        xml_path = annotations_dir / xml_file
        img_dir  = base_dir / angle / side / "bad" / cat

        if not xml_path.exists():
            print(f"[SKIP] Missing XML: {xml_path}")
            continue
        if not img_dir.exists():
            print(f"[SKIP] Missing img dir: {img_dir}")
            continue

        ann = load_xml_boxes(xml_path)
        chosen = None
        for name, gt_boxes in ann.items():
            img_path = img_dir / name
            if img_path.exists():
                chosen = (img_path, gt_boxes)
                break

        if chosen is None:
            print(f"[SKIP] No annotated images on disk for {cat}")
            continue

        img_path, gt_boxes = chosen
        print(f"[VIS] {cat}: {img_path.name}")
        # This call draws + saves side-by-side automatically
        detect_anomalies(str(img_path), str(mem_path), gt_boxes, cat)

# --------------------------------------------------------------
if __name__ == "__main__":
    pass
