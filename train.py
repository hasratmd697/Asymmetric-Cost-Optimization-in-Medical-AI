import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.models as models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ---- Reproducibility ----
SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything()

# ---- Configuration / Hyperparameters ----
BASE_DIR       = "./data"  # Path to competition data
IMG_DIR        = os.path.join(BASE_DIR, "images")
TRAIN_CSV      = os.path.join(BASE_DIR, "train.csv")
TEST_CSV       = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB     = os.path.join(BASE_DIR, "sample_submission.csv")

IMG_SIZE       = 256          # Resize target
BATCH_SIZE     = 64           # Batch size
NUM_WORKERS    = 4
NUM_EPOCHS     = 12
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
N_FOLDS        = 5            # Stratified K-fold
TRAIN_FOLDS    = [0, 1, 2]    # Train on 3 folds, validate on remaining 2
FN_WEIGHT      = 5.0          # Asymmetric False Negative penalty
FP_WEIGHT      = 1.0          # False Positive penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Dataset & Transforms ----
class ChestXrayDataset(Dataset):
    """Custom PyTorch Dataset for loading chest X-ray images."""
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform
        self.is_test   = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img_id = row["id"]
        img_path = os.path.join(self.img_dir, img_id)

        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_id
        else:
            label = int(row["label"])
            return image, label

# Data Augmentation (Train)
train_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation transforms
val_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test-Time Augmentation (TTA) transforms
tta_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- Model Definition ----
class ChestXrayModel(nn.Module):
    """DenseNet-121 backbone with custom classifier head."""
    def __init__(self, num_classes=20, pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()  # Remove original head

        # Custom classifier with dropout and batchnorm regularisation
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class BalancedCELoss(nn.Module):
    """Cross-entropy with SQRT inverse-frequency class weights + label smoothing."""
    def __init__(self, class_counts, num_classes, smoothing=0.05):
        super().__init__()
        total = sum(class_counts.values())
        weights = []
        for c in range(num_classes):
            freq = class_counts.get(c, 1)
            w = np.sqrt(total / (num_classes * freq))
            weights.append(w)
        w_tensor = torch.tensor(weights, dtype=torch.float32)
        w_tensor = w_tensor / w_tensor.sum() * num_classes  # Normalise
        self.register_buffer("weight", w_tensor)
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth * log_probs).sum(dim=1)
        w = self.weight[targets]
        loss = (loss * w).mean()
        return loss

# ---- Metrics & Training Hooks ----
def compute_competition_score(y_true, y_pred, num_classes=20):
    """Macro-averaged asymmetric cost score."""
    class_scores = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        n_c = (y_true == c).sum()
        if n_c == 0:
            continue
        score_c = (tp - fp - FN_WEIGHT * fn) / n_c
        class_scores.append(score_c)
    return np.mean(class_scores)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_score = compute_competition_score(np.array(all_labels), np.array(all_preds))
    return epoch_loss, epoch_score

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Valid", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_score = compute_competition_score(np.array(all_labels), np.array(all_preds))
    return epoch_loss, epoch_score, np.array(all_preds), np.array(all_labels)

# ---- Main Executable ----
def main():
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found. Please place training data in {BASE_DIR}/")
        return

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    
    CLASS_NAMES = [c for c in train_df.columns if c != "id"]
    NUM_CLASSES = len(CLASS_NAMES)
    train_df["label"] = train_df[CLASS_NAMES].values.argmax(axis=1)

    print(f"Dataset Loaded. Classes: {NUM_CLASSES}")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_models = []
    fold_val_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        if fold_idx not in TRAIN_FOLDS:
            continue
            
        print(f"\nTraining Fold {fold_idx + 1}/{N_FOLDS}...")
        df_train_fold = train_df.iloc[train_idx]
        df_val_fold   = train_df.iloc[val_idx]

        # Sqrt-balanced sampler
        labels_train = df_train_fold["label"].values
        class_sample_count = np.bincount(labels_train, minlength=NUM_CLASSES).astype(float)
        class_sample_count[class_sample_count == 0] = 1.0
        sample_weights = 1.0 / np.sqrt(class_sample_count[labels_train])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(df_train_fold), replacement=True)

        train_ds = ChestXrayDataset(df_train_fold, IMG_DIR, transform=train_transforms)
        val_ds   = ChestXrayDataset(df_val_fold,   IMG_DIR, transform=val_transforms)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model = ChestXrayModel(num_classes=NUM_CLASSES, pretrained=True, dropout=0.4).to(DEVICE)
        cc = dict(Counter(labels_train))
        criterion = BalancedCELoss(cc, NUM_CLASSES).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        scaler = GradScaler()

        best_score = -999
        best_state = None
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            train_loss, train_score = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
            val_loss, val_score, _, _ = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            print(f"  Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}")

            if val_score > best_score:
                best_score = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 6:
                    print("  ➜ Early stopping triggered.")
                    break

        model.load_state_dict(best_state)
        fold_models.append(model)
        fold_val_scores.append(best_score)
        
        # Save checkpoints locally
        torch.save(best_state, f"densenet121_fold{fold_idx}.pth")
        
        del train_ds, val_ds, train_loader, val_loader, optimizer, scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nMean CV Score: {np.mean(fold_val_scores):.4f}")

if __name__ == "__main__":
    main()
