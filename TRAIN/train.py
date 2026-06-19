
"""
train.py
========
Fonctions d'entraînement et de validation pour SegDinoRegressorV2.
 
Métrique du challenge :
  w_i    = 1/30 + GT_i
  Err    = sum(w_i * (p_i - GT_i)²) / sum(w_i)
  Score  = (Err_F + Err_M) / 2 + |Err_F - Err_M|
 
Convention genre dans le CSV : 0.0 = femme, 1.0 = homme
 
Fonctions publiques :
  train_one_epoch(...)  → (train_loss, train_challenge_score)
  validate(...)         → (val_loss, val_challenge_score, err_f, err_m)
  evaluate_full(...)    → dict complet
  save_checkpoint(...)
  load_checkpoint(...)
"""
 
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helper genre — gère 0.0/1.0 ET "M"/"F" ET "male"/"female"
# Convention CSV : 0.0 = femme, 1.0 = homme
# ─────────────────────────────────────────────────────────────────────────────
 
def _is_female(g) -> bool:
    s = str(g).strip().lower()
    return s in ("0", "0.0", "f", "female")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Calcul du score challenge (numpy)
# ─────────────────────────────────────────────────────────────────────────────
 
def _weighted_err_np(preds: np.ndarray, targets: np.ndarray) -> float:
    """Err = sum(w_i * (p_i - GT_i)²) / sum(w_i)  avec w_i = 1/30 + GT_i"""
    w = 1.0 / 30.0 + targets
    return float(np.sum(w * (preds - targets) ** 2) / np.sum(w))
 
 
def challenge_score_np(
    preds:   np.ndarray,
    targets: np.ndarray,
    genders: np.ndarray,
) -> tuple[float, float, float]:
    """
    Calcule le score officiel du challenge.
    Retourne (score, err_f, err_m)
    """
    mask_f = np.array([_is_female(g) for g in genders])
    mask_m = ~mask_f
 
    err_f = _weighted_err_np(preds[mask_f], targets[mask_f]) if mask_f.sum() > 0 \
            else _weighted_err_np(preds, targets)
    err_m = _weighted_err_np(preds[mask_m], targets[mask_m]) if mask_m.sum() > 0 \
            else _weighted_err_np(preds, targets)
 
    score = (err_f + err_m) / 2.0 + abs(err_f - err_m)
    return score, err_f, err_m
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Entraînement — une époque
# ─────────────────────────────────────────────────────────────────────────────
 
def train_one_epoch(
    model:     nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    torch.amp.GradScaler,
    scheduler,
    clip_grad: float = 1.0,
) -> tuple[float, float]:
    """
    Entraîne le modèle pour une époque complète.
    Retourne (loss_moyenne, challenge_score_époque)
    """
    model.train()
 
    if hasattr(model, "backbone"):
        model.backbone.eval()
        blocks = model._get_blocks() if hasattr(model, "_get_blocks") else []
        for block in blocks:
            if any(p.requires_grad for p in block.parameters()):
                block.train()
 
    total_loss = 0.0
    all_preds, all_targets, all_genders = [], [], []
    n_batches = len(loader)
 
    batch_bar = tqdm(loader, desc="  train", leave=False, unit="batch")
 
    for X, y, gender, filename in batch_bar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
 
        optimizer.zero_grad(set_to_none=True)
 
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            pred = model(X)
            loss = criterion(pred, y, gender)
 
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=clip_grad,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
 
        total_loss += loss.item()
        all_preds.extend(pred.detach().float().cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        all_genders.extend(gender)
 
        batch_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[-1]:.1e}",
        )
 
    score, err_f, err_m = challenge_score_np(
        np.array(all_preds), np.array(all_targets), np.array(all_genders)
    )
    return total_loss / n_batches, score
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
 
@torch.no_grad()
def validate(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float, float]:
    """
    Évalue le modèle sur le loader de validation.
    Retourne (loss_moyenne, challenge_score, err_f, err_m)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_genders = [], [], []
    n_batches = len(loader)
 
    batch_bar = tqdm(loader, desc="  val  ", leave=False, unit="batch")
 
    for X, y, gender, filename in batch_bar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
 
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            pred = model(X)
            loss = criterion(pred, y, gender)
 
        total_loss += loss.item()
        all_preds.extend(pred.float().cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        all_genders.extend(gender)
 
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")
 
    score, err_f, err_m = challenge_score_np(
        np.array(all_preds), np.array(all_targets), np.array(all_genders)
    )
    return total_loss / n_batches, score, err_f, err_m
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Évaluation détaillée
# ─────────────────────────────────────────────────────────────────────────────
 
@torch.no_grad()
def evaluate_full(
    model:  nn.Module,
    loader,
    device: torch.device,
) -> dict:
    """
    Évaluation complète — collecte prédictions, cibles, genres, noms de fichiers.
    Affiche toutes les métriques dont le score challenge officiel.
    """
    model.eval()
    all_preds, all_targets, all_genders, all_filenames = [], [], [], []
 
    for X, y, gender, filenames in tqdm(loader, desc="Évaluation", unit="batch"):
        X = X.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            preds = model(X)
        all_preds.extend(preds.float().cpu().numpy())
        all_targets.extend(y.numpy())
        all_genders.extend(gender)
        all_filenames.extend(filenames)
 
    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    genders = np.array(all_genders)
 
    score, err_f, err_m = challenge_score_np(preds, targets, genders)
    mae  = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
 
    mask_f = np.array([_is_female(g) for g in genders])
    mask_m = ~mask_f
    mae_f  = np.abs(preds[mask_f] - targets[mask_f]).mean() if mask_f.sum() > 0 else float("nan")
    mae_m  = np.abs(preds[mask_m] - targets[mask_m]).mean() if mask_m.sum() > 0 else float("nan")
 
    print(f"\n── Métriques ───────────────────────────────")
    print(f"  Score challenge : {score:.6f}  ← métrique officielle")
    print(f"  Err_F (femme)   : {err_f:.6f}  ({mask_f.sum()} samples)")
    print(f"  Err_M (homme)   : {err_m:.6f}  ({mask_m.sum()} samples)")
    print(f"  Pénalité genre  : {abs(err_f - err_m):.6f}")
    print(f"  MAE global      : {mae:.4f}")
    print(f"  MAE [F]         : {mae_f:.4f}")
    print(f"  MAE [M]         : {mae_m:.4f}")
    print(f"  RMSE            : {rmse:.4f}")
 
    return {
        "preds":           preds,
        "targets":         targets,
        "genders":         all_genders,
        "filenames":       all_filenames,
        "challenge_score": score,
        "err_f":           err_f,
        "err_m":           err_m,
        "mae":             mae,
        "rmse":            rmse,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
 
def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    val_score: float,
    path:      str,
) -> None:
    """Sauvegarde le modèle complet (backbone dégelé + têtes)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "val_score":  val_score,
    }, path)
 
 
def load_checkpoint(
    model:     nn.Module,
    path:      str,
    optimizer: torch.optim.Optimizer = None,
    scheduler                        = None,
    device:    torch.device          = torch.device("cpu"),
) -> dict:
    """Charge un checkpoint complet. Retourne le dict pour reprendre l'entraînement."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    score = ckpt.get("val_score", float("inf"))
    print(f"✓ Checkpoint chargé — epoch {ckpt.get('epoch', '?')}  val_score={score:.6f}")
    return ckpt