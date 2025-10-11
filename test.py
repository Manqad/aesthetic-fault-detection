# test.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rembg import remove

# Configuration
IMG_SIZE = 512
ANOMALY_THRESHOLD = 3.9  # Fixed threshold for raw anomaly scores
MIN_CONTOUR_AREA = 20  # Increased to filter noise
MAX_ASPECT_RATIO = 10.0  # Filter elongated contours

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
        self.memory_bank = None

    def extract_features(self, x):
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        print(f"Test layer2 shape: {x2.shape}, layer3 shape: {x3.shape}")
        x3 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        features = []
        for feat in [x2, x3]:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w).permute(0, 2, 1)
            features.append(feat)
        return torch.cat(features, dim=2)

    def compute_anomaly_score(self, features, device):
        if self.memory_bank is None:
            raise ValueError("Memory bank not loaded.")
        memory_bank = self.memory_bank.to(device)
        print(f"Memory bank shape: {memory_bank.shape}, features shape: {features.shape}")
        # Flatten memory bank to [total_patches, channels]
        memory_bank = memory_bank.view(-1, memory_bank.size(-1))
        # Ensure features is [1, num_patches, channels]
        features = features.view(1, -1, features.size(-1))
        distances = torch.cdist(features, memory_bank, p=2)
        min_distances, _ = distances.min(dim=2)  # Min over memory bank patches
        return min_distances.squeeze(0)  # [num_patches]

    def forward(self, x):
        return self.extract_features(x)

# Detect Anomalies
def detect_anomalies(image_path, memory_bank_path):
    """Detect anomalies in the defective image, draw bounding boxes on original image, and save checkpoint results."""
    model = PatchCore()
    model.memory_bank = torch.load(memory_bank_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create checkpoint directory
    image_name = os.path.basename(image_path).split('.')[0]
    checkpoint_dir = f"checkpoints/checkpoints_{image_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load original image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Remove background using rembg and get alpha channel
    image = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    image = remove(image)
    img_normalized = np.array(image.convert("RGB"))
    alpha_channel = np.array(image)[:, :, 3]  # Extract alpha channel (0=transparent, 255=opaque)

    # Apply preprocessing (no cropping)
    img_normalized = cv2.convertScaleAbs(img_normalized, beta=-20)
    img_normalized = cv2.bilateralFilter(img_normalized, 5, 20, 20)
    lab = cv2.cvtColor(img_normalized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab = cv2.merge((l_clahe, a, b))
    img_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Save preprocessed image
    preprocessed_path = os.path.join(checkpoint_dir, f"preprocessed_{image_name}.png")
    cv2.imwrite(preprocessed_path, cv2.cvtColor(img_normalized, cv2.COLOR_RGB2BGR))
    print(f"Preprocessed image saved to {preprocessed_path}")

    # Note: Uncomment the next line to skip preprocessing and use original image
    # img_normalized = orig_img

    # Display preprocessed image
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB))
    plt.title("Preprocessed Image (Background Removed)")
    plt.axis('off')
    plt.show()

    # Create binary mask from alpha channel
    alpha_mask = cv2.resize(alpha_channel, (img_normalized.shape[1], img_normalized.shape[0]), interpolation=cv2.INTER_NEAREST)
    alpha_mask = (alpha_mask > 0).astype(np.uint8)  # 1 for object, 0 for background

    img = Image.fromarray(cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.extract_features(img_tensor)
        print(f"Test features shape: {features.shape}")
        anomaly_scores = model.compute_anomaly_score(features, device)
        print(f"Anomaly scores shape: {anomaly_scores.shape}")

    # Generate anomaly map
    b, num_patches, _ = features.shape
    grid_h, grid_w = 64, 64
    if num_patches != grid_h * grid_w:
        raise ValueError(f"Expected {grid_h * grid_w} patches, got {num_patches}")
    anomaly_map = anomaly_scores.view(1, 1, grid_h, grid_w).cpu().numpy()
    anomaly_map = cv2.resize(anomaly_map[0, 0], (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply alpha mask to anomaly map
    anomaly_map = anomaly_map * alpha_mask  # Zero out background

    # Save and display raw anomaly map
    plt.figure(figsize=(10, 5))
    plt.imshow(anomaly_map, cmap='hot')
    plt.title("Raw Anomaly Map (Background Masked)")
    plt.colorbar()
    plt.axis('off')
    raw_anomaly_path = os.path.join(checkpoint_dir, f"raw_anomaly_map_{image_name}.png")
    plt.savefig(raw_anomaly_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    print(f"Raw anomaly map saved to {raw_anomaly_path}")

    # Threshold raw anomaly map
    anomaly_map_binary = (anomaly_map > ANOMALY_THRESHOLD).astype(np.uint8) * 255

    # Save and display binary anomaly map
    binary_anomaly_path = os.path.join(checkpoint_dir, f"binary_anomaly_map_{image_name}.png")
    cv2.imwrite(binary_anomaly_path, anomaly_map_binary)
    print(f"Binary anomaly map saved to {binary_anomaly_path}")
    plt.figure(figsize=(10, 5))
    plt.imshow(anomaly_map_binary, cmap='gray')
    plt.title(f"Binary Anomaly Map (Raw Scores > {ANOMALY_THRESHOLD})")
    plt.axis('off')
    plt.show()

    # Find contours in the binary anomaly map
    contours, _ = cv2.findContours(anomaly_map_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Number of contours detected (raw scores > {ANOMALY_THRESHOLD}): {len(contours)}")
    boxes = []
    anomaly_scores_list = []  # Renamed to avoid conflict with variable name
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = max(w/h, h/w) if h != 0 else float('inf')
        print(f"Contour size: w={w}, h={h}, area={area}, aspect_ratio={aspect_ratio:.2f}")
        # Filter small or elongated contours
        if area > MIN_CONTOUR_AREA and aspect_ratio < MAX_ASPECT_RATIO:
            boxes.append((x, y, x + w, y + h))
            region_score = anomaly_map[y:y+h, x:x+w].mean()
            anomaly_scores_list.append(region_score)

    anomaly_scores_list = np.array(anomaly_scores_list)
    print(f"Anomaly scores: {anomaly_scores_list}")

    if len(boxes) == 0:
        print(f"No anomalies detected with raw scores > {ANOMALY_THRESHOLD}.")
    else:
        print(f"Detected {len(boxes)} potential anomalies with raw scores > {ANOMALY_THRESHOLD}.")

    # Draw bounding boxes on original image
    result_img = orig_img.copy()
    for (box, score) in zip(boxes, anomaly_scores_list):
        x1, y1, x2, y2 = box
        label = f"Anomaly: {score:.2f}"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(result_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save and display annotated image
    output_image_path = os.path.join(checkpoint_dir, f"output_{image_name}.png")
    cv2.imwrite(output_image_path, result_img)
    print(f"Annotated image saved to {output_image_path}")
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image with Detected Anomalies (Raw Scores > {ANOMALY_THRESHOLD})")
    plt.axis('off')
    plt.show()

    # Save result summary
    output_result_path = os.path.join(checkpoint_dir, f"results_{image_name}.txt")
    with open(output_result_path, 'w') as f:
        f.write("Anomaly Detection Results\n")
        f.write("------------------------\n")
        f.write("Timestamp: 01:01 PM PKT, Thursday, July 31, 2025\n")  # Update timestamp as needed
        f.write("Image: " + os.path.basename(image_path) + "\n")
        f.write("------------------------\n")
        f.write("ID | X1 | Y1 | X2 | Y2 | Area | Confidence Score\n")
        f.write("-----------------------------------------------\n")
        for i, (box, score) in enumerate(zip(boxes, anomaly_scores_list), 1):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            f.write(f"{i}  | {x1} | {y1} | {x2} | {y2} | {area} | {score:.2f}\n")
    print(f"Result summary saved to {output_result_path}")

    return result_img