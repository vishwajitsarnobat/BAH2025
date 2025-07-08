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
import time
import json

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =================================================================================
# 1. CONFIGURATION (Static, driven by JSON)
# =================================================================================
class Config:
    DATA_PATH = "./cityscapes_data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 19
    IMG_HEIGHT = 256
    IMG_WIDTH = 512
    BATCH_SIZE = 4 # Adjust based on your GPU memory
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

# =================================================================================
# 2. DATA HANDLING MODULE
# =================================================================================
class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.cityscapes = Cityscapes(root, split=split, mode='fine', target_type='semantic')
        self.ignore_index = 255
        self.class_map = dict(zip([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33], range(Config.NUM_CLASSES)))
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
            'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]

    def __len__(self):
        return len(self.cityscapes)

    def _map_target(self, target):
        target_np = np.array(target)
        mapped_target = np.full(target_np.shape, self.ignore_index, dtype=np.uint8)
        for k, v in self.class_map.items():
            mapped_target[target_np == k] = v
        return torch.from_numpy(mapped_target).long()

    def __getitem__(self, idx):
        image, target = self.cityscapes[idx]
        image_np = np.array(image)
        target_np = np.array(target)

        if self.transform:
            augmented = self.transform(image=image_np, mask=target_np)
            image = augmented['image']
            target = self._map_target(augmented['mask'])
        else:
            image = image_np
            target = self._map_target(target_np)
        
        return image, target

def get_sampler(dataset_root):
    raw_dataset = CityscapesDataset(root=dataset_root, split='train')
    num_samples_for_weights = min(1000, len(raw_dataset))
    indices = np.random.choice(len(raw_dataset), num_samples_for_weights, replace=False)
    
    print("Calculating pixel-based class weights for sampler...")
    z = np.zeros(Config.NUM_CLASSES)
    for i in tqdm(indices, desc="Analyzing samples"):
        _, target = raw_dataset[i]
        mask = (target >= 0) & (target < Config.NUM_CLASSES)
        labels = target[mask].numpy()
        count = np.bincount(labels, minlength=Config.NUM_CLASSES)
        z += count
    
    class_weights = 1.0 / (z + 1e-6)
    sample_weights = np.zeros(len(raw_dataset))
    
    for i in tqdm(range(len(raw_dataset)), desc="Assigning weights"):
        _, target = raw_dataset[i]
        present_classes = np.unique(target.numpy())
        w_classes = present_classes[present_classes != 255]
        if len(w_classes) > 0:
            sample_weights[i] = class_weights[w_classes].max()
    
    return torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def get_data_loaders(exp_config):
    train_transform = A.Compose([
        A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH), A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
    ])

    train_dataset = CityscapesDataset(root=Config.DATA_PATH, split='train', transform=train_transform)
    val_dataset = CityscapesDataset(root=Config.DATA_PATH, split='val', transform=val_transform)

    train_sampler, shuffle = None, True
    if exp_config.get('sampler', False):
        train_sampler = get_sampler(Config.DATA_PATH)
        shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, sampler=train_sampler,
        shuffle=shuffle, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, val_loader

# =================================================================================
# 3. MODEL & LOSS FACTORY MODULE
# =================================================================================
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.focal = smp.losses.FocalLoss(mode='multiclass', gamma=gamma, ignore_index=255)
        self.lovasz = smp.losses.LovaszLoss(mode='multiclass', per_image=False, ignore_index=255)

    def forward(self, y_pred, y_true):
        return self.alpha * self.focal(y_pred, y_true) + self.beta * self.lovasz(y_pred, y_true)

def get_model_and_optimizer_from_exp(exp_config):
    arch = exp_config['architecture']
    encoder = exp_config['encoder']
    if arch == "unet": 
        model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", classes=Config.NUM_CLASSES)
    elif arch == "deeplabv3plus": 
        model = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet", classes=Config.NUM_CLASSES)
    elif arch == "pspnet": 
        model = smp.PSPNet(encoder_name=encoder, encoder_weights="imagenet", classes=Config.NUM_CLASSES)
    else: 
        raise ValueError(f"Unknown architecture: {arch}")
    model.to(Config.DEVICE)

    loss_choice = exp_config['loss']
    if loss_choice == "ce": 
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    elif loss_choice == "focal": 
        loss_fn = smp.losses.FocalLoss(mode='multiclass', ignore_index=255, gamma=2)
    elif loss_choice == "lovasz": 
        loss_fn = smp.losses.LovaszLoss(mode='multiclass', per_image=False, ignore_index=255) # CORRECTED: Added Lovasz
    elif loss_choice == "hybrid": 
        loss_fn = HybridLoss()
    else: 
        raise ValueError(f"Unknown loss: {loss_choice}")

    opt_choice = exp_config['optimizer']
    if opt_choice == "adam": 
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    elif opt_choice == "adamw": 
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    else: 
        raise ValueError(f"Unknown optimizer: {opt_choice}")

    print(f"Loaded: Arch={arch}, Encoder={encoder}, Loss={loss_choice}, Optimizer={opt_choice}, Sampler={exp_config['sampler']}")
    return model, loss_fn, optimizer

# =================================================================================
# 4. TRAINING & EVALUATION MODULE
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    for data, targets in loop:
        data, targets = data.to(device), targets.to(device)
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def evaluate_fn(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validating"):
            data = data.to(device)
            preds = torch.argmax(model(data), dim=1).cpu().numpy().flatten()
            labels = targets.cpu().numpy().flatten()
            mask = labels != 255
            all_preds.extend(preds[mask])
            all_labels.extend(labels[mask])

    cm = confusion_matrix(all_preds, all_preds, labels=list(range(Config.NUM_CLASSES)))
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / (union + 1e-6)
    mean_iou = np.nanmean(iou)
    return mean_iou, iou

# =================================================================================
# 5. EXPERIMENT MANAGER
# =================================================================================
class Tee(object):
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files: 
            f.flush()

def run_experiment(exp_config):
    log_dir, model_dir = "logs", "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, f"{exp_config['name_code']}_logs.txt")
    original_stdout = sys.stdout
    log_file = open(log_file_path, 'w')
    sys.stdout = Tee(original_stdout, log_file)

    try:
        print(f"--- Starting Experiment: {exp_config['name_pretty']} ---")
        print(f"Start time: {time.ctime()}")
        print(f"Full Config: {exp_config}\n")
        
        train_loader, val_loader = get_data_loaders(exp_config)
        model, loss_fn, optimizer = get_model_and_optimizer_from_exp(exp_config)
        
        best_iou = -1
        for epoch in range(Config.EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
            train_loss = train_fn(train_loader, model, optimizer, loss_fn, Config.DEVICE)
            print(f"Average Training Loss: {train_loss:.4f}")
            
            mean_iou, iou_per_class = evaluate_fn(val_loader, model, Config.DEVICE)
            
            print("\n--- Validation Metrics ---")
            print(f"Mean IoU (mIoU): {mean_iou:.4f}")
            class_names = val_loader.dataset.class_names
            for i, name in enumerate(class_names):
                print(f"  Class '{name}' IoU: {iou_per_class[i]:.4f}")
            print("--------------------------\n")

            if mean_iou > best_iou:
                best_iou = mean_iou
                torch.save(model.state_dict(), os.path.join(model_dir, f"{exp_config['name_code']}_best_model.pth"))
                print(f"=> Saved new best model with mIoU: {mean_iou:.4f}")
        
        print(f"\n--- Experiment {exp_config['name_pretty']} Finished ---")
        print(f"End time: {time.ctime()}")

    finally:
        sys.stdout = original_stdout
        log_file.close()

# =================================================================================
# 6. MAIN EXECUTION BLOCK
# =================================================================================
if __name__ == "__main__":
    if not os.path.exists(Config.DATA_PATH):
        print(f"Error: Dataset not found at '{Config.DATA_PATH}'")
        sys.exit(1)
        
    try:
        with open('experiments.json', 'r') as f:
            experiments_to_run = json.load(f)['experiments']
    except FileNotFoundError:
        print("Error: experiments.json not found. Please create it.")
        sys.exit(1)

    print(f"Found {len(experiments_to_run)} experiments to run.")
    
    for i, exp_config in enumerate(experiments_to_run):
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT {i+1}/{len(experiments_to_run)}: {exp_config['name_pretty']}")
        print(f"{'='*80}\n")
        run_experiment(exp_config)
        
    print("\n\nAll experiments completed! You can now run plot_results.py.")