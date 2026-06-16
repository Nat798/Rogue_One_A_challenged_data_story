# run_phase2.py
# Phase 2 — Fine-tuning sur dataset original pur depuis best_model.pth (phase 1).
# LR ÷10, sampler genre seul, early stopping patience 5.
# Charge les poids de phase 1 uniquement (pas l'optimizer/scheduler).

import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from multiprocessing import freeze_support

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

load_dotenv(override=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CSV_TRAIN, CSV_TEST, IMAGE_DIR, CHECKPOINT_PHASE1, SAVE_DIR_PHASE2

from model_dino import SegDinoRegressorV2, OcclusionLoss
from dataset    import FaceOcclusionDataset
from train      import train_one_epoch, validate, evaluate_full, save_checkpoint, load_checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
VAL_SIZE    = int(os.getenv("VAL_SIZE",    "20000"))
N_EPOCHS    = int(os.getenv("N_EPOCHS",    "30"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

MAX_LR_LORA     = float(os.getenv("MAX_LR_LORA",     "5e-7"))
MAX_LR_BACKBONE = float(os.getenv("MAX_LR_BACKBONE", "1e-6"))
MAX_LR_DECODER  = float(os.getenv("MAX_LR_DECODER",  "1e-5"))
WARMUP_EPOCHS   = int(os.getenv("WARMUP_EPOCHS", "2"))
CLIP_GRAD       = float(os.getenv("CLIP_GRAD",   "1.0"))
GENDER_PENALTY  = float(os.getenv("GENDER_PENALTY", "1.0"))

USE_LORA           = os.getenv("USE_LORA", "true").lower() == "true"
LORA_R             = int(os.getenv("LORA_R",     "32"))
LORA_ALPHA         = float(os.getenv("LORA_ALPHA", "64.0"))
N_UNFROZEN_BLOCKS  = int(os.getenv("N_UNFROZEN_BLOCKS", "2"))
N_PATCHES          = int(os.getenv("N_PATCHES",  "196"))
INTERMEDIATE_BLOCK = int(os.getenv("INTERMEDIATE_BLOCK", "5"))
DROPOUT            = float(os.getenv("DROPOUT", "0.2"))

RESUME_PATH             = os.getenv("RESUME_PATH", CHECKPOINT_PHASE1)
SAVE_DIR                = os.getenv("SAVE_DIR",    SAVE_DIR_PHASE2)
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "10"))
RANDOM_SEED             = 42


# ─────────────────────────────────────────────────────────────────────────────
# Sampler genre seul (sans rééquilibrage par bin d'occlusion)
# ─────────────────────────────────────────────────────────────────────────────

def make_gender_only_sampler(df, gender_col="gender"):
    """Équilibre uniquement F/M — laisse la distribution GT naturelle."""
    genders       = df[gender_col].astype(str).values
    n_total       = len(df)
    classes, cnts = np.unique(genders, return_counts=True)
    gender_w      = {c: n_total / (len(classes) * cnt) for c, cnt in zip(classes, cnts)}
    w             = np.array([gender_w[g] for g in genders], dtype=np.float32)
    print("Sampler genre seul (phase 2) :")
    for c, cnt in zip(classes, cnts):
        print(f"  Genre {c} : {cnt:>6} exemples  poids {gender_w[c]:.3f}")
    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=n_total, replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement dataset — reproduit le split exactement comme la phase 1
# ─────────────────────────────────────────────────────────────────────────────

def build_phase2_loaders(csv_train, csv_test, image_dir, val_size=20_000,
                         batch_size=32, num_workers=4, random_seed=42,
                         gender_col="gender", label_col="FaceOcclusion") -> dict:
    """Split train/val identique à la phase 1 (même seed). Sampler genre seul."""
    df_all  = pd.read_csv(csv_train, delimiter=",").dropna().sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_test = pd.read_csv(csv_test,  delimiter=",").dropna()

    df_all["occ_bin"]   = pd.qcut(df_all[label_col], q=10, labels=False, duplicates="drop")
    df_all["strat_key"] = df_all[gender_col].astype(str) + "_" + df_all["occ_bin"].astype(str)

    df_train_orig, df_val = train_test_split(
        df_all, test_size=val_size / len(df_all),
        stratify=df_all["strat_key"], random_state=random_seed,
    )
    df_train_orig = df_train_orig.drop(["occ_bin", "strat_key"], axis=1).reset_index(drop=True)
    df_val        = df_val.drop(       ["occ_bin", "strat_key"], axis=1).reset_index(drop=True)
    df_train_orig["_aug"] = False
    df_val["_aug"]        = False

    print(f"\nDataset phase 2 (original pur) :")
    print(f"  Train : {len(df_train_orig):,}  |  Val : {len(df_val):,}  |  Test : {len(df_test):,}")

    train_dataset = FaceOcclusionDataset(df_train_orig, image_dir, mode="train")
    val_dataset   = FaceOcclusionDataset(df_val,        image_dir, mode="val")
    test_dataset  = FaceOcclusionDataset(df_test,       image_dir, mode="test")

    train_sampler = make_gender_only_sampler(df_train_orig, gender_col=gender_col)

    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    return {
        "train":    DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, **common),
        "val":      DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **common),
        "test":     DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **common),
        "df_train": df_train_orig, "df_val": df_val, "df_test": df_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print("\n── Chargement des données (phase 2 — original pur) ─────────────────")
    data         = build_phase2_loaders(CSV_TRAIN, CSV_TEST, IMAGE_DIR, VAL_SIZE, BATCH_SIZE, NUM_WORKERS, RANDOM_SEED)
    train_loader = data["train"]
    val_loader   = data["val"]
    test_loader  = data["test"]

    print("\n── Initialisation du modèle ─────────────────────────────────────────")
    model = SegDinoRegressorV2(
        use_lora=USE_LORA, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
        n_unfrozen_blocks=N_UNFROZEN_BLOCKS, intermediate_block_idx=INTERMEDIATE_BLOCK,
        attn_block_idx=-1, n_patches=N_PATCHES, dropout=DROPOUT,
    ).to(device)

    if not os.path.isfile(RESUME_PATH):
        raise FileNotFoundError(f"Checkpoint phase 1 introuvable : {RESUME_PATH}")
    ckpt        = torch.load(RESUME_PATH, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    prev_score  = ckpt.get("val_score", float("inf"))
    prev_epoch  = ckpt.get("epoch", "?")
    print(f"\n✓ Poids chargés — epoch {prev_epoch}  val_score_phase1={prev_score:.6f}")
    if missing:    print(f"  ⚠ Clés manquantes  ({len(missing)})  : {missing[:5]}")
    if unexpected: print(f"  ⚠ Clés inattendues ({len(unexpected)}): {unexpected[:5]}")

    criterion    = OcclusionLoss(gender_penalty=GENDER_PENALTY)
    param_groups = model.get_trainable_param_groups(lr_lora=MAX_LR_LORA, lr_backbone=MAX_LR_BACKBONE, lr_decoder=MAX_LR_DECODER)
    optimizer    = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    total_steps  = len(train_loader) * N_EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    scheduler    = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[MAX_LR_LORA, MAX_LR_LORA, MAX_LR_BACKBONE, MAX_LR_DECODER],
        total_steps=total_steps, pct_start=warmup_steps / total_steps, anneal_strategy="cos",
    )
    scaler            = torch.amp.GradScaler(device.type, enabled=True)
    best_val_score    = prev_score
    epochs_no_improve = 0

    print(f"\n── Phase 2 : {N_EPOCHS} epochs max — early stopping patience {EARLY_STOPPING_PATIENCE} ──")
    history   = []
    epoch_bar = tqdm(range(N_EPOCHS), desc="Phase 2", unit="epoch")

    for epoch in epoch_bar:
        train_loss, train_score = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, scheduler, clip_grad=CLIP_GRAD)
        val_loss, val_score, err_f, err_m = validate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_score": train_score,
            "val_loss": val_loss, "val_score": val_score, "err_f": err_f, "err_m": err_m,
            "penalty": abs(err_f - err_m),
            "lr_lora": scheduler.get_last_lr()[0], "lr_backbone": scheduler.get_last_lr()[2], "lr_decoder": scheduler.get_last_lr()[3],
        })

        epoch_bar.set_postfix({"tr": f"{train_score:.5f}", "val": f"{val_score:.5f}",
                                "best": f"{best_val_score:.5f}", "F": f"{err_f:.5f}", "M": f"{err_m:.5f}",
                                "no_imp": f"{epochs_no_improve}/{EARLY_STOPPING_PATIENCE}"})

        save_checkpoint(model, optimizer, scheduler, epoch + 1, val_score, os.path.join(SAVE_DIR, "last_checkpoint.pth"))

        if val_score < best_val_score:
            best_val_score    = val_score
            epochs_no_improve = 0
            best_tmp  = os.path.join(SAVE_DIR, "best_model.tmp.pth")
            best_path = os.path.join(SAVE_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_score, best_tmp)
            os.replace(best_tmp, best_path)
            tqdm.write(f"  ✓ Best p2 epoch {epoch+1}  score={val_score:.6f}  err_F={err_f:.6f}  err_M={err_m:.6f}  [Δ vs p1 : {val_score - prev_score:+.6f}]")
        else:
            epochs_no_improve += 1
            tqdm.write(f"  · Epoch {epoch+1}  val={val_score:.6f}  pas d'amélioration ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")

        pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, "history_phase2.csv"), index=False)

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            tqdm.write(f"\n⚑ Early stopping — {EARLY_STOPPING_PATIENCE} epochs sans amélioration")
            break

    print(f"\n✓ Meilleur score phase 2 : {best_val_score:.6f}  (phase 1 : {prev_score:.6f}  Δ : {best_val_score - prev_score:+.6f})")

    best_ckpt = os.path.join(SAVE_DIR, "best_model.pth")
    if os.path.isfile(best_ckpt):
        load_checkpoint(model, best_ckpt, device=device)
    else:
        print(f"⚠ Aucun best model phase 2 — le best phase 1 reste la référence : {RESUME_PATH}")

    evaluate_full(model, val_loader, device)

    print("\n── Inférence test ───────────────────────────────────────────────────")
    model.eval()
    all_preds, all_filenames = [], []
    with torch.no_grad():
        for X, filenames in tqdm(test_loader, desc="Test"):
            X = X.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                preds = model(X)
            all_preds.extend(preds.float().cpu().numpy())
            all_filenames.extend(filenames)

    submission = pd.DataFrame({"filename": all_filenames, "FaceOcclusion": all_preds})
    sub_path   = os.path.join(SAVE_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"✓ Soumission : {sub_path}")


if __name__ == "__main__":
    freeze_support()
    main()
