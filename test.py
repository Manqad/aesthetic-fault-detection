# test.py — Raw-distance thresholding + FG mask + NMS + visualization
from pyexpat import model
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
IMG_SIZE = 1024
ANOMALY_THRESHOLD = 4.25  # tune higher to reduce FPs, lower to catch more

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

COMPARISON_DIR = Path(r'C:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\comparisons')
COMPARISON_DIR.mkdir(exist_ok=True)
SAVED_CATEGORIES = {"bad1": False, "bad2": False, "bad3": False}

# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------
def nms_xyxy(boxes, scores, iou_thr=0.3):
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
    if not BACKGROUND_REMOVAL or not _HAS_REMBG:
        if BACKGROUND_REMOVAL and not _HAS_REMBG:
            print("[WARN] BACKGROUND_REMOVAL=True but rembg not available. Proceeding without masking.")
        w, h = pil_rgb.size
        return np.ones((h, w), dtype=np.uint8) * 255
    rgba = pil_rgb.convert("RGBA")
    rgba_removed = rembg_remove(rgba)
    alpha = np.array(rgba_removed)[:, :, 3]
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

        # Your custom truncation
        self.layer2 = nn.Sequential(*list(self.backbone.children())[:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[:7])
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

    def compute_anomaly_score(self, features, device, chunk=8192):
        if self.memory_bank is None:
            raise ValueError("Memory bank not loaded.")
        bank = self.memory_bank
        if bank.dim() == 3:
            bank = bank.view(-1, bank.size(-1))
        bank = bank.float().cpu()
        feat = features.view(-1, features.size(-1)).float().cpu()
        mins = []
        for i in range(0, feat.size(0), chunk):
            f = feat[i:i+chunk]
            d = torch.cdist(f, bank, p=2)
            mins.append(d.min(dim=1).values)
        return torch.cat(mins, dim=0)

    def forward(self, x):
        return self.extract_features(x)

# --------------------------------------------------------------
# SAVE COMPARISON
# --------------------------------------------------------------
def save_comparison(img_pred, img_gt, img_name, cat):
    comparison = np.hstack([img_pred, img_gt])
    save_path = COMPARISON_DIR / f"comparison_{cat}_{Path(img_name).stem}.jpg"
    cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"  SAVED: {save_path}")

# --------------------------------------------------------------
# DETECT ANOMALIES
# --------------------------------------------------------------
def detect_anomalies(img_path: str, memory_bank_path: str, gt_boxes: list, cat: str):
    global SAVED_CATEGORIES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCoreInference().to(device)
    model.eval()

    model.memory_bank = torch.load(memory_bank_path, map_location="cpu", weights_only=True)
    if model.memory_bank.dim() == 3:
        model.memory_bank = model.memory_bank.view(-1, model.memory_bank.size(-1))

    pil_rgb = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_rgb.size
    fg_mask = make_foreground_mask(pil_rgb)
    fg_mask_bool = (fg_mask > 0).astype(np.uint8)

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = tf(pil_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(tensor)
        anomaly_scores = model.compute_anomaly_score(features, device).numpy()

    # ----------------------------------------------------------
    # FIXED: auto-detect patchmap size (no longer hard-coded 64)
    # ----------------------------------------------------------
    N = len(anomaly_scores)
    side = int(np.sqrt(N))
    if side * side != N:
        raise ValueError(f"[ERROR] Unexpected anomaly_scores size {N}. Cannot reshape into square.")
    score_map = anomaly_scores.reshape(side, side)
    # ----------------------------------------------------------

    score_map = cv2.resize(score_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    masked_score = score_map.copy()
    masked_score[fg_mask_bool == 0] = 0.0
    binary = (masked_score > ANOMALY_THRESHOLD).astype(np.uint8) * 255

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept_xyxy, kept_scores = [], []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl, 0], stats[lbl, 1], stats[lbl, 2], stats[lbl, 3], stats[lbl, 4]
        if w < MIN_BOX_W or h < MIN_BOX_H: continue
        if area < MIN_CONTOUR_AREA:        continue
        aspect = max(w / h, h / w) if min(w, h) > 0 else float("inf")
        if aspect > MAX_ASPECT_RATIO:      continue

        x1_t, y1_t, x2_t, y2_t = x, y, x + w - 1, y + h - 1
        x1_t = max(0, x1_t); y1_t = max(0, y1_t)
        x2_t = min(x2_t, orig_w - 1); y2_t = min(y2_t, orig_h - 1)
        if (x2_t - x1_t) > 6 and (y2_t - y1_t) > 6:
            x1_t += 1; y1_t += 1; x2_t -= 1; y2_t -= 1

        m = box_mean_score(score_map, x1_t, y1_t, x2_t, y2_t)
        kept_xyxy.append([x1_t, y1_t, x2_t, y2_t])
        kept_scores.append(m)

    keep_idx = nms_xyxy(kept_xyxy, kept_scores, iou_thr=NMS_IOU)
    pred_boxes_xyxy = [kept_xyxy[i] for i in keep_idx]

    pred_boxes_eval = [[x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)] for (x1, y1, x2, y2) in pred_boxes_xyxy]

    img_pred = np.array(pil_rgb)
    for (x1, y1, x2, y2) in pred_boxes_xyxy:
        cv2.rectangle(img_pred, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    img_gt = np.array(pil_rgb)
    for (x, y, w, h) in gt_boxes:
        x1, y1 = int(x), int(y); x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 0, 255), 3)

    if not SAVED_CATEGORIES.get(cat, False):
        save_comparison(img_pred, img_gt, img_path, cat)
        SAVED_CATEGORIES[cat] = True

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

def evaluate_detection(pred, gt, iou_thr=0.2, debug=False, img_name=None, image_size=None, alpha=0.5):
    """
    Hybrid evaluation:
      - IoU F1 (instance matching)
      - Area F1 (coverage)
      - Combined Hybrid F1 = α*IoU + (1−α)*Area
    """
    # --- IoU-based ---
    if not pred and not gt:
        iou_prec = iou_rec = iou_f1 = 1.0
    elif not pred or not gt:
        iou_prec = iou_rec = iou_f1 = 0.0
    else:
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
        iou_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        iou_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou_f1   = 2 * iou_prec * iou_rec / (iou_prec + iou_rec) if (iou_prec + iou_rec) > 0 else 0.0

    # --- Area-based ---
    if image_size is not None:
        area_res = evaluate_detection_area(pred, gt, image_size=image_size)
        area_f1 = area_res["area_f1"]
    else:
        area_f1 = 0.0

    # --- Hybrid blend ---
    hybrid_f1_val = hybrid_f1(iou_f1, area_f1, alpha=alpha)

    return {
        "precision": iou_prec,
        "recall": iou_rec,
        "f1_score": iou_f1,
        "area_f1": area_f1,
        "hybrid_f1": hybrid_f1_val,
    }


# ==============================================================
# NEW: ADDITIONAL EVALUATION MODES (NO CHANGE TO DETECTION)
# ==============================================================

# -------------------------------
# Area-based (pixel overlap) F1
# -------------------------------
def _rasterize_boxes_xywh(boxes, image_size):
    """
    Convert [x,y,w,h] boxes into a binary mask (H,W).
    Compatible with your +1 width/height convention.
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x, y, w, h) in boxes:
        x1 = int(np.floor(x)); y1 = int(np.floor(y))
        x2 = int(np.ceil(x + w)); y2 = int(np.ceil(y + h))
        x1 = max(0, min(W, x1)); y1 = max(0, min(H, y1))
        x2 = max(0, min(W, x2)); y2 = max(0, min(H, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask

def evaluate_detection_area(pred, gt, image_size, debug=False, img_name=None):
    """
    Pixel-level Coverage/Purity F1 using unions (symmetric, merge/split tolerant).
    Returns: {"area_precision","area_recall","area_f1"}
    """
    H, W = image_size
    if H <= 0 or W <= 0:
        return {"area_precision": 0.0, "area_recall": 0.0, "area_f1": 0.0}

    pred_mask = _rasterize_boxes_xywh(pred, (H, W))
    gt_mask   = _rasterize_boxes_xywh(gt, (H, W))

    pred_area = int(pred_mask.sum())
    gt_area   = int(gt_mask.sum())
    inter     = int(np.logical_and(pred_mask, gt_mask).sum())

    if pred_area == 0 and gt_area == 0:
        return {"area_precision": 1.0, "area_recall": 1.0, "area_f1": 1.0}
    if pred_area == 0 or gt_area == 0:
        return {"area_precision": 0.0, "area_recall": 0.0, "area_f1": 0.0}

    area_precision = inter / pred_area
    area_recall    = inter / gt_area
    denom = (area_precision + area_recall)
    area_f1        = 2 * area_precision * area_recall / denom if denom > 0 else 0.0

    if debug:
        tag = f" [AREA-DBG {img_name}]" if img_name else " [AREA-DBG]"
        print(f"{tag} inter={inter} pred_area={pred_area} gt_area={gt_area} "
              f"P:{area_precision:.3f} R:{area_recall:.3f} F1:{area_f1:.3f}")

    return {"area_precision": area_precision, "area_recall": area_recall, "area_f1": area_f1}

# -----------------------------------------
# Group-IoU (merge/split-aware, symmetric)
# -----------------------------------------
def _mask_iou(m1, m2):
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    return (inter / union) if union > 0 else 0.0, inter, union

def _build_overlap_graph(pred, gt, image_size, iou_eps=0.05):
    """Bipartite graph preds <-> gts using small mask-IoU threshold."""
    H, W = image_size
    pred_masks = [_rasterize_boxes_xywh([p], (H, W)) for p in pred]
    gt_masks   = [_rasterize_boxes_xywh([g], (H, W)) for g in gt]

    pred_adj = [set() for _ in range(len(pred))]
    gt_adj   = [set() for _ in range(len(gt))]

    for pi, pm in enumerate(pred_masks):
        for gi, gm in enumerate(gt_masks):
            iou_pg, _, _ = _mask_iou(pm, gm)
            if iou_pg > iou_eps:
                pred_adj[pi].add(gi)
                gt_adj[gi].add(pi)

    return pred_masks, gt_masks, pred_adj, gt_adj

def _connected_components(pred_adj, gt_adj):
    """Connected components over bipartite graph; returns list of (Pset, Gset)."""
    from collections import deque
    nP, nG = len(pred_adj), len(gt_adj)
    visitedP, visitedG = [False]*nP, [False]*nG
    comps = []

    for startP in range(nP):
        if visitedP[startP]:
            continue
        if len(pred_adj[startP]) == 0:
            visitedP[startP] = True
            comps.append(({startP}, set()))
            continue
        qP = deque([startP])
        curP, curG = set(), set()
        visitedP[startP] = True
        while qP:
            p = qP.popleft()
            curP.add(p)
            for g in pred_adj[p]:
                if not visitedG[g]:
                    visitedG[g] = True
                    curG.add(g)
                    for pp in gt_adj[g]:
                        if not visitedP[pp]:
                            visitedP[pp] = True
                            qP.append(pp)
        comps.append((curP, curG))

    for g in range(nG):
        if not visitedG[g] and len(gt_adj[g]) == 0:
            visitedG[g] = True
            comps.append((set(), {g}))
    return comps

def evaluate_detection_group(pred, gt, image_size, iou_eps=0.05, debug=False, img_name=None):
    """
    Symmetric, merge/split-aware scoring via Group-IoU over overlap components.
    Returns: {"group_precision","group_recall","group_f1","num_components"}
    """
    H, W = image_size
    if not pred and not gt:
        return {"group_precision": 1.0, "group_recall": 1.0, "group_f1": 1.0, "num_components": 0}
    if H <= 0 or W <= 0:
        return {"group_precision": 0.0, "group_recall": 0.0, "group_f1": 0.0, "num_components": 0}

    pred_masks, gt_masks, pred_adj, gt_adj = _build_overlap_graph(pred, gt, (H, W), iou_eps=iou_eps)
    comps = _connected_components(pred_adj, gt_adj)

    TP_soft = 0.0; FP_soft = 0.0; FN_soft = 0.0
    for (Pset, Gset) in comps:
        if len(Pset) > 0:
            U_pred = np.zeros((H, W), dtype=np.uint8)
            for pi in Pset: U_pred |= pred_masks[pi]
        else:
            U_pred = np.zeros((H, W), dtype=np.uint8)
        if len(Gset) > 0:
            U_gt = np.zeros((H, W), dtype=np.uint8)
            for gi in Gset: U_gt |= gt_masks[gi]
        else:
            U_gt = np.zeros((H, W), dtype=np.uint8)

        gIoU, inter, union = _mask_iou(U_pred, U_gt)
        TP_soft += gIoU
        FP_soft += max(0.0, len(Pset) - gIoU)
        FN_soft += max(0.0, len(Gset) - gIoU)

        if debug:
            tag = f"[GROUP-DBG {img_name}]" if img_name else "[GROUP-DBG]"
            print(f"{tag} comp(P={sorted(list(Pset))}, G={sorted(list(Gset))}) "
                  f"inter={inter} union={union} gIoU={gIoU:.3f}  "
                  f"FP+= {max(0.0, len(Pset) - gIoU):.2f}  FN+= {max(0.0, len(Gset) - gIoU):.2f}")

    prec = TP_soft / (TP_soft + FP_soft) if (TP_soft + FP_soft) > 0 else 0.0
    rec  = TP_soft / (TP_soft + FN_soft) if (TP_soft + FN_soft) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"group_precision": prec, "group_recall": rec, "group_f1": f1, "num_components": len(comps)}

# -----------------------
# Optional: Hybrid blend
# -----------------------
def hybrid_f1(iou_f1: float, area_f1: float, alpha: float = 0.5) -> float:
    """Weighted blend of IoU-F1 (instance strictness) and Area-F1 (coverage)."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * iou_f1 + (1.0 - alpha) * area_f1

# --------------------------------------------------------------
# POST-TEST VISUALIZATION (unchanged)
# --------------------------------------------------------------
def visualize_after_testing(angle: str,
                            side: str,
                            base_dir: str | Path,
                            annotations_dir: str | Path,
                            memory_bank_dir: str | Path,
                            categories=None):
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
        detect_anomalies(str(img_path), str(mem_path), gt_boxes, cat)

# --------------------------------------------------------------
if __name__ == "__main__":
    pass
