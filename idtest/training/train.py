"""Minimal training loop: Adam + CrossEntropy + best-val-AUC checkpoint.

Generic enough to reuse for other self-trained detectors later. Designed for
small data (no workers, no scheduler) — a proof-of-concept, not production.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def _evaluate(model, loader, device):
    """ROC-AUC of P(fake)=softmax[:,1] over a labeled loader."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(device))
            logits = out[0] if isinstance(out, tuple) else out   # Xception returns (logits, feat)
            pfake = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            ys.extend(labels.numpy().tolist())
            ss.extend(pfake.tolist())
    ys = np.asarray(ys)
    ss = np.asarray(ss)
    if np.unique(ys).size < 2:
        return float("nan")
    return float(roc_auc_score(ys, ss))


def train_model(
    model,
    train_paths,
    val_paths,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    save_path: str,
    size: int = 256,
    seed: int = 0,
) -> float:
    """Train ``model`` on FWA reals+synthetic-fakes; checkpoint best val AUC.

    Returns the best validation AUC achieved. ``val_paths`` are held-out reals
    (their synthetic fakes are generated on the fly too).
    """
    from idtest.training.dataset import FWADataset

    torch.manual_seed(seed)
    train_loader = DataLoader(
        FWADataset(train_paths, size=size, do_augment=True, seed=seed),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        FWADataset(val_paths, size=size, do_augment=False, seed=seed + 1),
        batch_size=batch_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best = -1.0
    for epoch in range(epochs):
        model.train()
        running, nbatch = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out   # Xception returns (logits, feat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            nbatch += 1
        auc = _evaluate(model, val_loader, device)
        print(f"[train] epoch {epoch + 1}/{epochs}  loss={running / max(1, nbatch):.4f}  val_auc={auc:.4f}")
        if auc == auc and auc > best:          # not NaN and improved
            best = auc
            torch.save(model.state_dict(), save_path)
    print(f"[train] best val_auc={best:.4f}; checkpoint -> {save_path}")
    return best
