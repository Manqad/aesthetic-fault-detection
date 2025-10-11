# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 512
BATCH_SIZE = 4
DATA_DIR = "data/front_flipped"  # Update this to your local data directory
MEMORY_BANK_PATH = "patchcore_memory_bank.pth"
CORESET_SAMPLING_RATIO = 0.7  # Note: This ratio is high; consider lowering to 0.01-0.1 for typical PatchCore usage
PATCH_SIZE = 4  # Not used in code, but kept for reference

# Custom Dataset
class PerfectPartsDataset(Dataset):
    """Dataset for loading preprocessed images."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(root, f))
        if not self.image_paths:
            raise ValueError(f"No .jpg, .png, or .jpeg images found in {root_dir} or its subdirectories.")
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            raise ValueError(f"Error loading {img_path}: {e}")

# PatchCore Model
class PatchCore(nn.Module):
    """PatchCore model for anomaly detection with patch-wise feature extraction."""
    def __init__(self, backbone_name="wide_resnet50_2"):
        super(PatchCore, self).__init__()
        self.backbone = models.__dict__[backbone_name](weights="IMAGENET1K_V1")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.layer2 = nn.Sequential(*list(self.backbone.children())[:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[:7])
        self.memory_bank = []

    def extract_features(self, x):
        x2 = self.layer2(x)  # Shape: [batch_size, 512, 64, 64]
        x3 = self.layer3(x)  # Shape: [batch_size, 1024, 32, 32]
        print(f"layer2 shape: {x2.shape}, layer3 shape: {x3.shape}")
        x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        features = []
        for feat in [x2, x3]:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w).permute(0, 2, 1)
            features.append(feat)
        return torch.cat(features, dim=2)  # [batch_size, 4096, 1536]

    def add_to_memory_bank(self, features):
        self.memory_bank.append(features.detach().cpu())

    def coreset_sampling(self):
        if not self.memory_bank:
            return
        memory_bank = torch.cat(self.memory_bank, dim=0)
        # Fix: Flatten to patch level for proper coreset sampling (common in PatchCore)
        # Original code sampled images; this samples patches for better representation
        memory_bank = memory_bank.view(-1, memory_bank.size(-1))  # [num_images * 4096, 1536]
        print(f"Memory bank size before sampling: {memory_bank.shape}")
        num_samples = max(1, int(CORESET_SAMPLING_RATIO * memory_bank.shape[0]))
        indices = torch.randperm(memory_bank.shape[0])[:num_samples]
        self.memory_bank = memory_bank[indices]
        print(f"Memory bank size after sampling: {self.memory_bank.shape}")

    def forward(self, x):
        return self.extract_features(x)

# Train PatchCore
def train_patchcore(data_dir):
    """Train PatchCore by building a memory bank of normal patch features."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        dataset = PerfectPartsDataset(root_dir=data_dir, transform=transform)
    except ValueError as e:
        print(f"Error: {e}")
        raise

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCore().to(device)
    model.eval()

    print("Extracting patch features for memory bank...")
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            features = model.extract_features(batch)
        model.add_to_memory_bank(features)
        print(f"Processed batch {i+1}/{len(dataloader)}, features shape: {features.shape}")

    print("Performing coreset sampling...")
    model.coreset_sampling()
    torch.save(model.memory_bank, MEMORY_BANK_PATH)
    print(f"Memory bank saved to {MEMORY_BANK_PATH}")

    # Visualize a sample image
    sample_img = dataset[0]
    sample_img_np = sample_img.permute(1, 2, 0).numpy()
    sample_img_np = (sample_img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
    plt.figure(figsize=(10, 5))
    plt.imshow(sample_img_np)
    plt.title("Sample Training Image")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train_patchcore(DATA_DIR)