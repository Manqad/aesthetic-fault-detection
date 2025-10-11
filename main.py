# main.py
from train import train_patchcore
from test import detect_anomalies

# Set your paths here
TRAIN_DATA_DIR = "data/front_flipped"  # Update this to your local training data directory
TEST_IMAGE_PATH = "data/faulty/front_flipped.JPG"  # Update this to your local test image path
MEMORY_BANK_PATH = "patchcore_memory_bank.pth"  # Path where memory bank will be saved/loaded

if __name__ == "__main__":
    # Train the model and save memory bank
    train_patchcore(TRAIN_DATA_DIR, MEMORY_BANK_PATH)
    
    # Test on the image using the saved memory bank
    result = detect_anomalies(TEST_IMAGE_PATH, MEMORY_BANK_PATH)
    print("Anomaly detection completed.")