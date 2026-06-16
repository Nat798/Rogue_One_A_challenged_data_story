# INFEROTRON_FINAL.py
# Inférence finale avec TTA (6 passes géométriques) + ensemble multi-checkpoints.
# Pas de flip horizontal : l'occlusion faciale est asymétrique → biais non annulable.
# Produit : TRAIN/submission_tta.csv

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CSV_TEST, IMAGE_DIR, SUBMISSION_CSV, CHECKPOINT_PHASE2

from model_dino import SegDinoRegressorV2

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINTS = [
    CHECKPOINT_PHASE2,
    # Ajouter d'autres checkpoints pour l'ensemble :
    # os.path.join(os.path.dirname(__file__), "runs", "vitb_explora_aug", "best_model.pth"),
]

OUTPUT_CSV = SUBMISSION_CSV
BATCH_SIZE = 32
TTA_N      = 6

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# Transforms TTA (6 passes — sans flip horizontal)
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TTA_TRANSFORMS = [
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(_MEAN, _STD)]),
    T.Compose([T.Resize(288), T.CenterCrop(224), T.ToTensor(), T.Normalize(_MEAN, _STD)]),
    T.Compose([T.Resize(256), T.CenterCrop(200), T.Resize(224), T.ToTensor(), T.Normalize(_MEAN, _STD)]),
    T.Compose([T.Resize(288), T.CenterCrop(224), T.ColorJitter(brightness=0.1), T.ToTensor(), T.Normalize(_MEAN, _STD)]),
    T.Compose([T.Resize(256), T.CenterCrop(200), T.Resize(224), T.ColorJitter(brightness=0.1), T.ToTensor(), T.Normalize(_MEAN, _STD)]),
    T.Compose([T.Resize(256), T.CenterCrop(224), T.ColorJitter(contrast=0.1), T.ToTensor(), T.Normalize(_MEAN, _STD)]),
]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset test
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    """Retourne un stack de toutes les versions TTA pour chaque image."""
    def __init__(self, filenames, image_root, tta_n=1):
        self.filenames  = filenames
        self.image_root = image_root
        self.transforms = _TTA_TRANSFORMS[:tta_n] if tta_n > 1 else [_TTA_TRANSFORMS[0]]
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        rel_path = self.filenames[idx]
        img      = Image.open(os.path.join(self.image_root, rel_path)).convert("RGB")
        tensors  = torch.stack([t(img) for t in self.transforms])
        return tensors, rel_path


# ─────────────────────────────────────────────────────────────────────────────
# Chargement modèle
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> torch.nn.Module:
    print(f"\nChargement : {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model = SegDinoRegressorV2(
        use_lora=True, lora_r=32, lora_alpha=64.0, n_unfrozen_blocks=2,
        intermediate_block_idx=5, attn_block_idx=-1, n_patches=196, dropout=0.0,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        non_trivial = [k for k in missing if "weight" in k and "lora" not in k and "LayerNorm" not in k and "norm" not in k]
        if non_trivial:
            print(f"  ⚠ Clés manquantes non-triviales ({len(non_trivial)}) : {non_trivial[:5]}")
    model.eval()
    print(f"  ✓ Epoch {ckpt.get('epoch','?')}  val_score={ckpt.get('val_score', float('nan')):.6f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inférence avec TTA
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_tta(model, loader, tta_n, desc="Inférence") -> np.ndarray:
    """Retourne un array (N,) de prédictions moyennées sur tta_n passes."""
    all_preds = []
    for X_tta, _ in tqdm(loader, desc=desc, unit="batch"):
        B           = X_tta.shape[0]
        batch_preds = torch.zeros(B, device=device)
        for t in range(tta_n):
            X = X_tta[:, t].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                preds = model(X)
            batch_preds += preds.float()
        batch_preds /= tta_n
        all_preds.extend(batch_preds.cpu().numpy())
    return np.array(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    valid_checkpoints = [c for c in CHECKPOINTS if os.path.isfile(c)]
    if not valid_checkpoints:
        raise FileNotFoundError(f"Aucun checkpoint trouvé : {CHECKPOINTS}")

    print(f"\n{'='*60}")
    print(f"Inférence — TTA×{TTA_N} | Ensemble×{len(valid_checkpoints)}")
    print(f"{'='*60}")

    df_test   = pd.read_csv(CSV_TEST).dropna()
    filenames = df_test["filename"].tolist()
    print(f"  {len(filenames):,} images  |  TTA×{TTA_N}  |  {len(valid_checkpoints)} checkpoint(s)")

    dataset = TestDataset(filenames, IMAGE_DIR, tta_n=TTA_N)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    ensemble_preds = np.zeros(len(filenames))
    for i, ckpt_path in enumerate(valid_checkpoints):
        model = load_model(ckpt_path)
        preds = predict_with_tta(model, loader, tta_n=TTA_N, desc=f"Ckpt {i+1}/{len(valid_checkpoints)}")
        ensemble_preds += preds
        del model
        torch.cuda.empty_cache()

    ensemble_preds /= len(valid_checkpoints)
    ensemble_preds  = np.clip(ensemble_preds, 0.0, 1.0)

    print(f"\n── Stats prédictions ────────────────────────────────────────────────")
    print(f"  Min={ensemble_preds.min():.4f}  Max={ensemble_preds.max():.4f}  Mean={ensemble_preds.mean():.4f}  Median={np.median(ensemble_preds):.4f}")

    df_out = pd.DataFrame({"filename": filenames, "FaceOcclusion": ensemble_preds, "gender": "x"})
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ {len(df_out):,} prédictions → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
