"""
Training utilities for EEGNet and ShallowConvNet.
Supports within-subject cross-validation and leave-one-out evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def to_tensor(X, y, device):
    return (
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.long).to(device),
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_loss / total, correct / total, preds, labels


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    n_epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    device: str = "cpu",
    verbose: bool = True,
):
    """Train a model with early stopping. Returns best val accuracy."""
    device = torch.device(device)
    model = model.to(device)

    X_tr_t, y_tr_t = to_tensor(X_train, y_train, device)
    X_val_t, y_val_t = to_tensor(X_val, y_val, device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    iterator = tqdm(range(n_epochs), desc="Training", disable=not verbose)
    for epoch in iterator:
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            iterator.set_postfix(
                tr_acc=f"{tr_acc:.3f}", val_acc=f"{val_acc:.3f}", best=f"{best_val_acc:.3f}"
            )

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc, history


def cross_validate_model(
    model_fn,
    X,
    y,
    n_splits: int = 5,
    device: str = "cpu",
    **train_kwargs,
):
    """
    Stratified k-fold cross-validation for a neural model.
    model_fn: callable that returns a fresh model instance.
    Returns per-fold accuracies and predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs = []
    all_preds = np.zeros(len(y), dtype=np.int64)
    all_labels = np.zeros(len(y), dtype=np.int64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = model_fn()
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val_fold = y[train_idx], y[val_idx]

        best_acc, _ = train_model(
            model, X_tr, y_tr, X_val, y_val_fold, device=device, verbose=False, **train_kwargs
        )

        # Get final predictions on val fold
        model.eval()
        device_t = torch.device(device)
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device_t)
            preds = model(X_val_t).argmax(1).cpu().numpy()

        all_preds[val_idx] = preds
        all_labels[val_idx] = y_val_fold
        fold_accs.append(best_acc)
        print(f"  Fold {fold+1}/{n_splits} — val acc: {best_acc:.4f}")

    return fold_accs, all_preds, all_labels
