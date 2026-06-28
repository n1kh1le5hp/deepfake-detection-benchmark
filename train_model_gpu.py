import os
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 2e-4
NUM_WORKERS = 8
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS    = 1

DFDC_TRAIN_FAKE = "datasets/DFDC/train/fake"
DFDC_TRAIN_REAL = "datasets/DFDC/train/real"
DFDC_VAL_FAKE   = "datasets/DFDC/validation/fake"
DFDC_VAL_REAL   = "datasets/DFDC/validation/real"
MANIFEST_CSV    = "data/out/manifest.csv"
OUTPUT_DIR      = "outputs"


class ImageFolderDataset(Dataset):
    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        path, label = self.entries[i]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        return self.transform(img), label


def load_folder(folder, label):
    folder = Path(folder)
    return [(str(p), label) for p in sorted(folder.iterdir()) if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]


def load_manifest(csv_path):
    df = pd.read_csv(csv_path)
    return [(row['path'], 1 if row['split'] == 'fake' else 0) for _, row in df.iterrows()]


def build_model():
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
    if NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs with DataParallel")
        model = nn.DataParallel(model)
    return model.to(DEVICE)


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels, total_loss, correct, total = [], [], 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, labels)
            probs = torch.softmax(out, dim=1)[:, 1]
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), correct / total, auc, all_labels, all_probs


def make_loader(entries, transform, shuffle):
    dataset = ImageFolderDataset(entries, transform)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        multiprocessing_context='spawn' if NUM_WORKERS > 0 else None,
    )


def main():
    print(f"Device: {DEVICE} | GPUs: {NUM_GPUS} | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    for i in range(NUM_GPUS):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} ({p.total_memory // 1024**3}GB)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_entries = load_folder(DFDC_TRAIN_FAKE, 1) + load_folder(DFDC_TRAIN_REAL, 0)
    val_entries   = load_folder(DFDC_VAL_FAKE, 1)   + load_folder(DFDC_VAL_REAL, 0)
    test_entries  = load_manifest(MANIFEST_CSV)

    print(f"\nTrain — fake: {sum(1 for _,l in train_entries if l==1)}, real: {sum(1 for _,l in train_entries if l==0)}, total: {len(train_entries)}")
    print(f"Val   — fake: {sum(1 for _,l in val_entries   if l==1)}, real: {sum(1 for _,l in val_entries   if l==0)}, total: {len(val_entries)}")
    print(f"Test  — fake: {sum(1 for _,l in test_entries  if l==1)}, real: {sum(1 for _,l in test_entries  if l==0)}, total: {len(test_entries)}\n")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_loader = make_loader(train_entries, train_tf, shuffle=True)
    val_loader   = make_loader(val_entries,   eval_tf,  shuffle=False)
    test_loader  = make_loader(test_entries,  eval_tf,  shuffle=False)

    model     = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda')

    best_auc = 0
    history  = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            state = model.module.state_dict() if NUM_GPUS > 1 else model.state_dict()
            torch.save(state, f"{OUTPUT_DIR}/best_model.pth")
            print(f"  -> saved best model (AUC={best_auc:.4f})")

    print(f"\nBest Val AUC: {best_auc:.4f}")

    print("\nEvaluating on ID test set (manifest.csv) ...")
    base_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2).to(DEVICE)
    base_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth"))
    if NUM_GPUS > 1:
        base_model = nn.DataParallel(base_model)

    test_loss, test_acc, test_auc, test_labels, test_probs = evaluate(base_model, test_loader)
    print(f"\nID Test Set — Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")

    cm = confusion_matrix(test_labels, [1 if p > 0.5 else 0 for p in test_probs])
    print(f"Confusion Matrix:\n{cm}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curves')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='Val AUC', color='green')
    plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.legend(); plt.title('Validation AUC')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150)
    print(f"Curves saved to {OUTPUT_DIR}/training_curves.png")


if __name__ == "__main__":
    main()