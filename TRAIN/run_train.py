# run_train.py
# Entraînement phase 1 — ExPLoRA + dataset augmenté.
# Bascule vers le dataset original seul à partir de SWITCH_TO_ORIG_EPOCH.

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from multiprocessing import freeze_support

load_dotenv(override=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CSV_TRAIN, CSV_TEST, CSV_AUG, IMAGE_DIR, IMAGE_DIR_AUG, SAVE_DIR_PHASE1

from model_dino import SegDinoRegressorV2, OcclusionLoss
from dataset    import build_dataloaders
from train      import train_one_epoch, validate, evaluate_full, save_checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
VAL_SIZE      = int(os.getenv("VAL_SIZE",    "20000"))
N_EPOCHS      = int(os.getenv("N_EPOCHS",    "100"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE",  "32"))
NUM_WORKERS   = int(os.getenv("NUM_WORKERS", "4"))

MAX_LR_LORA     = float(os.getenv("MAX_LR_LORA",     "5e-6"))
MAX_LR_BACKBONE = float(os.getenv("MAX_LR_BACKBONE", "1e-5"))
MAX_LR_DECODER  = float(os.getenv("MAX_LR_DECODER",  "1e-4"))
WARMUP_EPOCHS   = int(os.getenv("WARMUP_EPOCHS", "5"))
CLIP_GRAD       = float(os.getenv("CLIP_GRAD",   "1.0"))

SWITCH_TO_ORIG_EPOCH = int(os.getenv("SWITCH_TO_ORIG_EPOCH", "36"))

GENDER_PENALTY = float(os.getenv("GENDER_PENALTY", "1"))

USE_LORA           = os.getenv("USE_LORA", "true").lower() == "true"
LORA_R             = int(os.getenv("LORA_R",     "32"))
LORA_ALPHA         = float(os.getenv("LORA_ALPHA", "64.0"))
N_UNFROZEN_BLOCKS  = int(os.getenv("N_UNFROZEN_BLOCKS", "2"))
N_PATCHES          = int(os.getenv("N_PATCHES",  "196"))
INTERMEDIATE_BLOCK = int(os.getenv("INTERMEDIATE_BLOCK", "5"))
DROPOUT            = float(os.getenv("DROPOUT", "0.2"))
BALANCE_GENDER     = os.getenv("BALANCE_GENDER", "true").lower() == "true"

SAVE_DIR    = os.getenv("SAVE_DIR",    SAVE_DIR_PHASE1)
RESUME_PATH = os.getenv("RESUME_PATH", "")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("\n── Chargement des données ───────────────────────────────────────────")
    data = build_dataloaders(
        csv_train=CSV_TRAIN, csv_test=CSV_TEST, image_dir=IMAGE_DIR,
        csv_aug=CSV_AUG, image_dir_aug=IMAGE_DIR_AUG,
        val_size=VAL_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        balance_gender=BALANCE_GENDER,
    )
    train_loader      = data["train"]
    train_loader_orig = data["train_orig"]
    val_loader        = data["val"]
    test_loader       = data["test"]

    print(f"\n── Bascule dataset original seul à l'epoch {SWITCH_TO_ORIG_EPOCH} ─────────────────")

    print("\n── Initialisation du modèle ─────────────────────────────────────────")
    model = SegDinoRegressorV2(
        use_lora=USE_LORA, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
        n_unfrozen_blocks=N_UNFROZEN_BLOCKS, intermediate_block_idx=INTERMEDIATE_BLOCK,
        attn_block_idx=-1, n_patches=N_PATCHES, dropout=DROPOUT,
    ).to(device)

    criterion    = OcclusionLoss(gender_penalty=GENDER_PENALTY)
    param_groups = model.get_trainable_param_groups(lr_lora=MAX_LR_LORA, lr_backbone=MAX_LR_BACKBONE, lr_decoder=MAX_LR_DECODER)
    optimizer    = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    total_steps  = len(train_loader) * N_EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    scheduler    = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = [MAX_LR_LORA, MAX_LR_LORA, MAX_LR_BACKBONE, MAX_LR_DECODER],
        total_steps     = total_steps,
        pct_start       = warmup_steps / total_steps,
        anneal_strategy = "cos",
    )
    scaler         = torch.amp.GradScaler(device.type, enabled=True)
    start_epoch    = 0
    best_val_score = float("inf")

    if RESUME_PATH and os.path.isfile(RESUME_PATH):
        from train import load_checkpoint
        ckpt           = load_checkpoint(model, RESUME_PATH, optimizer, scheduler, device)
        start_epoch    = ckpt.get("epoch", 0)
        best_val_score = ckpt.get("val_score", float("inf"))

    print(f"\n── Entraînement ({N_EPOCHS} epochs) ──────────────────────────────────────")
    history   = []
    epoch_bar = tqdm(range(start_epoch, N_EPOCHS), desc="Entraînement", unit="epoch")

    for epoch in epoch_bar:
        if (epoch + 1) >= SWITCH_TO_ORIG_EPOCH and train_loader is not train_loader_orig:
            tqdm.write(f"  → Epoch {epoch+1} : passage au dataset original seul")
            train_loader = train_loader_orig

        train_loss, train_score = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, scheduler, clip_grad=CLIP_GRAD,
        )
        val_loss, val_score, err_f, err_m = validate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_score": train_score,
            "val_loss": val_loss, "val_score": val_score, "err_f": err_f, "err_m": err_m,
            "penalty": abs(err_f - err_m),
            "lr_backbone": scheduler.get_last_lr()[0], "lr_decoder": scheduler.get_last_lr()[1],
        })

        epoch_bar.set_postfix({
            "tr_score": f"{train_score:.5f}", "val_score": f"{val_score:.5f}",
            "best": f"{best_val_score:.5f}", "err_F": f"{err_f:.5f}", "err_M": f"{err_m:.5f}",
        })

        save_checkpoint(model, optimizer, scheduler, epoch + 1, val_score, os.path.join(SAVE_DIR, "last_checkpoint.pth"))

        if val_score < best_val_score:
            best_val_score = val_score
            best_tmp  = os.path.join(SAVE_DIR, "best_model.tmp.pth")
            best_path = os.path.join(SAVE_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_score, best_tmp)
            os.replace(best_tmp, best_path)
            if (epoch + 1) % 5 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, val_score,
                                os.path.join(SAVE_DIR, f"best_epoch_{epoch+1:03d}.pth"))
            tqdm.write(f"  ✓ Best epoch {epoch+1}  score={val_score:.6f}  err_F={err_f:.6f}  err_M={err_m:.6f}")

        pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, "history.csv"), index=False)

    print(f"\n✓ Meilleur score : {best_val_score:.6f}")

    best_ckpt = os.path.join(SAVE_DIR, "best_model.pth")
    if os.path.isfile(best_ckpt):
        from train import load_checkpoint
        load_checkpoint(model, best_ckpt, device=device)
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
