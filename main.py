# --- Step 0: Install necessary packages ---
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install segmentation-models-pytorch albumentations==1.3.1 opencv-python-headless scikit-learn tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
import os
import sys

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
class Config:
    # IMPORTANT: Update this path to the folder containing 'gtFine' and 'leftImg8bit'
    DATA_PATH = "./cityscape_data" 
    
    MODEL_NAME = "deeplabv3plus"
    ENCODER = "efficientnet-b4"
    NUM_CLASSES = 19  # Cityscapes has 19 evaluation classes
    IMG_HEIGHT = 256
    IMG_WIDTH = 512
    BATCH_SIZE = 2
    EPOCHS = 10 # For a real experiment, use 50-100
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- The main switch: 'baseline' or 'advanced' ---
    RUN_MODE = "baseline" 

# --- Data Handling: Custom Dataset to map Cityscapes labels ---
class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.cityscapes = Cityscapes(root, split=split, mode='fine', target_type='semantic')
        
        self.ignore_index = 255
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(Config.NUM_CLASSES)))
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 
            'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]

    def __len__(self):
        return len(self.cityscapes)

    def _map_target(self, target):
        target = np.array(target)
        mapped_target = np.full(target.shape, self.ignore_index, dtype=np.uint8)
        for k, v in self.class_map.items():
            mapped_target[target == k] = v
        return torch.from_numpy(mapped_target).long()

    def __getitem__(self, idx):
        image, target = self.cityscapes[idx]
        image, target_np = np.array(image), np.array(target)

        if self.transform:
            augmented = self.transform(image=image, mask=target_np)
            image, target_np = augmented['image'], augmented['mask']
        
        target = self._map_target(target_np)
        return image, target

# --- ADVANCED: Component 1 - Class-Balanced Sampling ---
def get_sampler(dataset):
    print("Creating weighted sampler...")
    # First, calculate class weights based on pixel frequency
    print("Step 1: Calculating pixel-based class weights...")
    num_samples_for_weights = min(500, len(dataset))
    indices = np.random.choice(len(dataset), num_samples_for_weights, replace=False)
    z = np.zeros(Config.NUM_CLASSES)
    for i in tqdm(indices, desc="Analyzing samples for weights"):
        _, target = dataset[i]
        mask = (target >= 0) & (target < Config.NUM_CLASSES)
        labels = target[mask].numpy()
        count = np.bincount(labels, minlength=Config.NUM_CLASSES)
        z += count
    
    class_weights = 1.0 / (z + 1e-6) # Inverse frequency
    
    # Second, assign a weight to each image based on the classes it contains
    print("Step 2: Assigning weights to each image...")
    sample_weights = np.zeros(len(dataset))
    for i in tqdm(range(len(dataset)), desc="Weighting dataset"):
        _, target = dataset[i]
        present_classes, _ = np.unique(target, return_counts=True)
        # Weight of a sample is the max weight of any class present in it
        w = class_weights[present_classes[present_classes != 255]].max()
        sample_weights[i] = w
    
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

# --- ADVANCED: Component 2 - Hybrid Loss Function ---
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        # CORRECT: Initialize the losses with the ignore_index parameter.
        # This tells them to automatically ignore any pixel with the label 255.
        self.focal = smp.losses.FocalLoss(mode='multiclass', gamma=gamma, ignore_index=255)
        self.lovasz = smp.losses.LovaszLoss(mode='multiclass', per_image=False, ignore_index=255)
        
        print(f"HybridLoss initialized with alpha={alpha} (Focal) and beta={beta} (Lovasz), ignoring index 255.")

    def forward(self, y_pred, y_true):
        # CORRECT: The forward pass is now much simpler.
        # We no longer need to manually create and apply a mask.
        # The loss functions will handle the ignored pixels internally.
        
        focal_loss = self.focal(y_pred, y_true)
        lovasz_loss = self.lovasz(y_pred, y_true)
        
        return self.alpha * focal_loss + self.beta * lovasz_loss

# --- Model, Training, and Evaluation Functions ---
def get_model():
    print(f"Loading model: {Config.MODEL_NAME} with encoder {Config.ENCODER}")
    model = smp.create_model(
        Config.MODEL_NAME, encoder_name=Config.ENCODER, encoder_weights="imagenet",
        in_channels=3, classes=Config.NUM_CLASSES
    )
    return model.to(Config.DEVICE)

def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    for data, targets in loop:
        data, targets = data.to(Config.DEVICE), targets.to(Config.DEVICE)
        
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def evaluate_fn(loader, model, dataset):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validating"):
            data = data.to(Config.DEVICE)
            predictions = model(data)
            preds = torch.argmax(predictions, dim=1).cpu().numpy().flatten()
            labels = targets.cpu().numpy().flatten()
            
            mask = labels != 255
            all_preds.extend(preds[mask])
            all_labels.extend(labels[mask])

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(Config.NUM_CLASSES)))
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / (union + 1e-6)
    mean_iou = np.nanmean(iou)
    
    print("\n--- Validation Metrics ---")
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")
    for i, name in enumerate(dataset.class_names):
        print(f"  Class '{name}' IoU: {iou[i]:.4f}")
    print("--------------------------\n")
    return mean_iou

# --- Main Execution ---
def main():
    # Verify dataset path first
    if not os.path.exists(Config.DATA_PATH) or \
       not os.path.exists(os.path.join(Config.DATA_PATH, "leftImg8bit")) or \
       not os.path.exists(os.path.join(Config.DATA_PATH, "gtFine")):
        print(f"Error: Dataset not found or in wrong format at '{Config.DATA_PATH}'")
        print("Please download 'leftImg8bit_trainvaltest.zip' and 'gtFine_trainvaltest.zip',")
        print("unzip them into a single folder, and set DATA_PATH in the script.")
        sys.exit(1)

    print(f"Starting run in '{Config.RUN_MODE}' mode on device '{Config.DEVICE}'")

    train_transform = A.Compose([
        A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH), A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    train_dataset = CityscapesDataset(root=Config.DATA_PATH, split='train', transform=train_transform)
    val_dataset = CityscapesDataset(root=Config.DATA_PATH, split='val', transform=val_transform)
    
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # --- THE SWITCH: Select setup based on RUN_MODE ---
    if Config.RUN_MODE == 'baseline':
        print("\n--- Using Baseline Setup ---")
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        train_sampler = None
        shuffle = True
    
    elif Config.RUN_MODE == 'advanced':
        print("\n--- Using Advanced Setup ---")
        train_sampler = get_sampler(train_dataset)
        loss_fn = HybridLoss(alpha=0.5, beta=0.5)
        shuffle = False # Sampler and shuffle are mutually exclusive
    else:
        raise ValueError("Config.RUN_MODE must be 'baseline' or 'advanced'")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, sampler=train_sampler,
        shuffle=shuffle, num_workers=2, pin_memory=True,    
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    best_iou = -1
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        print(f"Average Training Loss: {train_loss:.4f}")
        
        iou = evaluate_fn(val_loader, model, val_dataset)
        
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), f"best_model_{Config.RUN_MODE}.pth")
            print(f"=> Saved new best model with mIoU: {iou:.4f}")

if __name__ == "__main__":
    main()