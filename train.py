# train.py — builds PatchCore memory bank (training phase)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
IMG_SIZE = 512
CORESET_CAP = 60000       # maximum patches to keep in memory bank
BATCH_SIZE = 4

# --------------------------------------------------------------
# PATCHCORE TRAINER
# --------------------------------------------------------------
class PatchCoreTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.layer2 = nn.Sequential(*list(self.backbone.children())[:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[:7])

    @torch.no_grad()
    def extract_features(self, x):
        """Return [B,4096,1536] patch embeddings"""
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        feats = []
        for feat in [x2, x3]:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h*w).permute(0, 2, 1)
            feats.append(feat)
        return torch.cat(feats, dim=2)

# --------------------------------------------------------------
# TRAIN FUNCTION
# --------------------------------------------------------------
def train_patchcore(good_dir, memory_bank_path):
    """
    Builds a PatchCore memory bank from all good images.
    Args:
        good_dir (str): path to good images directory
        memory_bank_path (str): output .pth file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PatchCoreTrainer().to(device)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img_paths = list(Path(good_dir).glob("*.jpg")) + \
                list(Path(good_dir).glob("*.png")) + \
                list(Path(good_dir).glob("*.jpeg"))
    print(f"Loaded {len(img_paths)} good images from {good_dir}")
    if len(img_paths) == 0:
        raise RuntimeError("No good images found!")

    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), BATCH_SIZE)):
            batch_files = img_paths[i:i+BATCH_SIZE]
            batch_imgs = []
            for f in batch_files:
                img = Image.open(f).convert("RGB")
                batch_imgs.append(transform(img))
            batch_tensor = torch.stack(batch_imgs).to(device)
            feats = trainer.extract_features(batch_tensor)  # [B,4096,1536]
            all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0)  # [N,4096,1536]
    n_patches = all_feats.shape[0] * all_feats.shape[1]
    print(f"Building streaming coreset on cpu... total patches: {n_patches:,}")

    # Flatten
    all_feats = all_feats.view(-1, all_feats.shape[-1])
    all_feats = all_feats.half()  # save space

    # Randomly sample up to CORESET_CAP
    if all_feats.shape[0] > CORESET_CAP:
        idx = torch.randperm(all_feats.shape[0])[:CORESET_CAP]
        all_feats = all_feats[idx]

    print(f"Final memory bank shape: {tuple(all_feats.shape)} (dtype={all_feats.dtype})")
    torch.save(all_feats, memory_bank_path)
    print(f"Memory bank saved to: {memory_bank_path}")
    return memory_bank_path

# --------------------------------------------------------------
if __name__ == "__main__":
    # example usage
    train_patchcore(
        good_dir="d:/fyp/aesthetic-fault-detection/Updated Dataset/acpart2/0º/back/good",
        memory_bank_path="d:/fyp/aesthetic-fault-detection/models/patchcore_0º_back.pth"
    )
