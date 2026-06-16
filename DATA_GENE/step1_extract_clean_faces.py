# step1_extract_clean_faces.py
# Extrait les visages non-occultés depuis CelebAMask-HQ.
# Double filtrage : structurel (masques hat/eye_g) puis modèle (pred < seuil).
# Produit : DATA/occlusions/clean_faces.csv

import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import (
    CELEB_IMG_DIR, CELEB_MASK_DIR, CELEB_ATTR_FILE,
    CSV_CLEAN_FACES, TRAIN_DIR,
)

sys.path.insert(0, TRAIN_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PTH           = os.path.join(TRAIN_DIR, "runs", "vitb_explora_aug", "best_model.pth")
OCCLUSION_THRESHOLD = 0.02
BATCH_SIZE          = 32
NUM_WORKERS         = 0

OCCLUSION_CLASSES = ["eye_g", "hat"]

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Filtrage structurel par masques
# ─────────────────────────────────────────────────────────────────────────────
def _subfolder_for(img_id: int) -> str:
    return str(img_id // 2000)

def _mask_prefix(img_id: int) -> str:
    return f"{img_id:05d}"

def mask_is_nonempty(mask_path: str) -> bool:
    if not os.path.isfile(mask_path):
        return False
    try:
        arr = np.array(Image.open(mask_path))
        return arr.max() > 0
    except Exception:
        return False

def structural_filter(n_total: int = 30000) -> list:
    print("\n── Étape 1 : filtrage structurel ────────────────────────────────────")
    clean_ids = []
    for img_id in tqdm(range(n_total), desc="Filtrage masques", unit="img"):
        folder   = os.path.join(CELEB_MASK_DIR, _subfolder_for(img_id))
        prefix   = _mask_prefix(img_id)
        occluded = any(
            mask_is_nonempty(os.path.join(folder, f"{prefix}_{cls}.png"))
            for cls in OCCLUSION_CLASSES
        )
        if not occluded:
            clean_ids.append(img_id)
    print(f"  Images sans occlusion : {len(clean_ids)} / {n_total}")
    return clean_ids

# ─────────────────────────────────────────────────────────────────────────────
# Chargement genre
# ─────────────────────────────────────────────────────────────────────────────
def load_gender_map() -> dict:
    print(f"\n── Chargement attributs genre ───────────────────────────────────────")
    if not os.path.isfile(CELEB_ATTR_FILE):
        print(f"  ⚠ Attributs introuvables : {CELEB_ATTR_FILE}")
        return {}
    with open(CELEB_ATTR_FILE, "r") as f:
        lines = f.readlines()
    headers  = lines[1].strip().split()
    if "Male" not in headers:
        return {}
    male_col   = headers.index("Male")
    gender_map = {}
    for line in lines[2:]:
        parts  = line.strip().split()
        if not parts:
            continue
        img_id = int(os.path.splitext(parts[0])[0])
        gender_map[img_id] = "1" if int(parts[male_col + 1]) == 1 else "0"
    print(f"  {len(gender_map)} entrées chargées")
    return gender_map

# ─────────────────────────────────────────────────────────────────────────────
# Filtrage modèle
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

VAL_TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=_MEAN, std=_STD),
])

class CelebCleanDataset(Dataset):
    def __init__(self, img_ids, img_dir):
        self.img_ids = img_ids
        self.img_dir = img_dir
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img    = Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
        return VAL_TRANSFORM(img), img_id

def load_model(pth_path: str) -> torch.nn.Module:
    from model_dino import SegDinoRegressorV2
    print(f"\n── Chargement modèle ────────────────────────────────────────────────")
    ckpt  = torch.load(pth_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model = SegDinoRegressorV2(
        use_lora=True, lora_r=32, lora_alpha=64.0,
        n_unfrozen_blocks=2, intermediate_block_idx=5,
        attn_block_idx=-1, n_patches=196, dropout=0.2,
    ).to(device)
    model.load_state_dict(state, strict=False)
    print(f"  ✓ Epoch {ckpt.get('epoch','?')}  val_score={ckpt.get('val_score', float('nan')):.6f}")
    model.eval()
    return model

@torch.no_grad()
def model_filter(model, img_ids, img_dir, threshold):
    print(f"\n── Étape 2 : filtrage modèle (seuil={threshold*100:.1f}%) ───────────")
    loader = DataLoader(
        CelebCleanDataset(img_ids, img_dir),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    all_ids, all_preds = [], []
    for X, ids in tqdm(loader, desc="Inférence modèle", unit="batch"):
        X = X.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            preds = model(X)
        all_ids.extend(ids.tolist())
        all_preds.extend(preds.float().cpu().numpy().tolist())
    all_ids   = np.array(all_ids)
    all_preds = np.array(all_preds)
    mask      = all_preds < threshold
    print(f"  Avant : {len(all_ids)} | Après : {mask.sum()}")
    return all_ids[mask].tolist(), all_preds[mask].tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for path, name in [(CELEB_IMG_DIR, "CelebA-HQ-img"), (CELEB_MASK_DIR, "mask-anno"), (MODEL_PTH, "best_model.pth")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Introuvable : {name}\n  → {path}")

    clean_ids_struct          = structural_filter(n_total=30000)
    gender_map                = load_gender_map()
    model                     = load_model(MODEL_PTH)
    clean_ids_model, clean_preds = model_filter(model, clean_ids_struct, CELEB_IMG_DIR, OCCLUSION_THRESHOLD)

    rows = [
        {
            "img_id":         img_id,
            "image_path":     os.path.join(CELEB_IMG_DIR, f"{img_id}.jpg"),
            "gender":         gender_map.get(img_id, "unknown"),
            "pred_occlusion": round(pred, 6),
        }
        for img_id, pred in zip(clean_ids_model, clean_preds)
    ]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_CLEAN_FACES), exist_ok=True)
    df.to_csv(CSV_CLEAN_FACES, index=False)
    print(f"\n✓ Sauvegardé : {CSV_CLEAN_FACES}  ({len(df)} images)")

if __name__ == "__main__":
    main()
