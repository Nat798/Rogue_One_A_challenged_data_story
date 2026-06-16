# dataset.py
# Chargement et préparation des données pour la régression d'occlusion faciale.
# Transforms train/val, sampler combiné genre×occlusion, intégration data_aug.csv.
# build_dataloaders retourne train (orig+aug), train_orig (orig seul), val, test.

import os
import io
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CSV_TRAIN, CSV_TEST, CSV_AUG, IMAGE_DIR, IMAGE_DIR_AUG

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation DINOv3 (identique ImageNet)
# ─────────────────────────────────────────────────────────────────────────────
_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Transforms custom
# ─────────────────────────────────────────────────────────────────────────────

class RandomJPEGCompression:
    """Simule des artefacts de compression JPEG."""
    def __init__(self, quality_range=(5, 40), p=0.3):
        self.quality_range = quality_range; self.p = p
    def __call__(self, img):
        if random.random() > self.p: return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.quality_range))
        buf.seek(0)
        return Image.open(buf).copy()

class RandomGaussianNoise:
    """Bruit gaussien (simule bruit capteur)."""
    def __init__(self, std_range=(0.02, 0.15), p=0.4):
        self.std_range = std_range; self.p = p
    def __call__(self, img):
        if random.random() > self.p: return img
        arr   = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, random.uniform(*self.std_range), arr.shape)
        return Image.fromarray((np.clip(arr + noise, 0, 1) * 255).astype(np.uint8))

class RandomLowResolution:
    """Downscale puis upscale : simule basse résolution."""
    def __init__(self, scale_range=(0.2, 0.5), p=0.3):
        self.scale_range = scale_range; self.p = p
    def __call__(self, img):
        if random.random() > self.p: return img
        w, h  = img.size
        scale = random.uniform(*self.scale_range)
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)

class RandomSaltPepperNoise:
    """Pixels corrompus sel & poivre."""
    def __init__(self, amount_range=(0.01, 0.05), p=0.2):
        self.amount_range = amount_range; self.p = p
    def __call__(self, img):
        if random.random() > self.p: return img
        arr    = np.array(img).copy()
        amount = random.uniform(*self.amount_range)
        n      = int(amount * arr.shape[0] * arr.shape[1])
        for val in [255, 0]:
            arr[np.random.randint(0, arr.shape[0], n), np.random.randint(0, arr.shape[1], n)] = val
        return Image.fromarray(arr)

class RandomHistogramEqualization:
    """Égalisation histogramme (simule scan/photo ancienne)."""
    def __init__(self, p=0.2):
        self.p = p
    def __call__(self, img):
        if random.random() > self.p: return img
        return ImageOps.equalize(img.convert("L")).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

VAL_TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=_DINO_MEAN, std=_DINO_STD),
])

TRAIN_TRANSFORM = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.20),
    T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.3),
    RandomGaussianNoise(std_range=(0.01, 0.06), p=0.30),
    RandomJPEGCompression(quality_range=(20, 50), p=0.20),
    RandomLowResolution(scale_range=(0.4, 0.7), p=0.15),
    RandomSaltPepperNoise(amount_range=(0.003, 0.02), p=0.10),
    T.ToTensor(),
    T.Normalize(mean=_DINO_MEAN, std=_DINO_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FaceOcclusionDataset(Dataset):
    def __init__(self, df, image_dir, image_dir_aug="", mode="train"):
        assert mode in ("train", "val", "test")
        self.df            = df.reset_index(drop=True)
        self.image_dir     = image_dir
        self.image_dir_aug = image_dir_aug if image_dir_aug else image_dir
        self.mode          = mode
        self.is_test       = (mode == "test")
        self.transform     = VAL_TRANSFORM if mode in ("val", "test") else TRAIN_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row      = self.df.loc[index]
        filename = row["filename"]
        is_aug   = bool(row.get("_aug", False))
        img_dir  = self.image_dir_aug if is_aug else self.image_dir
        img      = Image.open(os.path.join(img_dir, filename)).convert("RGB")
        X        = self.transform(img)
        if self.is_test:
            return X, filename
        return X, np.float32(row["FaceOcclusion"]), str(row["gender"]), filename


# ─────────────────────────────────────────────────────────────────────────────
# Sampler combiné genre × occlusion
# ─────────────────────────────────────────────────────────────────────────────

def make_combined_sampler(df, gender_col="gender", occ_col="FaceOcclusion", n_occ_bins=6):
    """Poids combiné genre × niveau d'occlusion — équilibre les batchs sur les deux dimensions."""
    genders = df[gender_col].astype(str).values
    occ     = df[occ_col].values
    n_total = len(df)

    classes, cnts = np.unique(genders, return_counts=True)
    gender_w      = {c: n_total / (len(classes) * cnt) for c, cnt in zip(classes, cnts)}
    w_gender      = np.array([gender_w[g] for g in genders], dtype=np.float32)

    occ_bins   = pd.qcut(pd.Series(occ), q=n_occ_bins, labels=False, duplicates="drop").fillna(0).astype(int).values
    n_bins_eff = occ_bins.max() + 1
    bin_counts = np.bincount(occ_bins, minlength=n_bins_eff).clip(min=1)
    w_occ      = (n_total / (n_bins_eff * bin_counts[occ_bins])).astype(np.float32)

    w_final = w_gender * w_occ
    w_final = w_final / w_final.mean()

    print("Sampler combiné genre × occlusion :")
    for c, cnt in zip(classes, cnts):
        print(f"  Genre {c} : {cnt} exemples  poids {gender_w[c]:.3f}")

    return WeightedRandomSampler(weights=torch.from_numpy(w_final), num_samples=n_total, replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Rééchantillonnage du dataset supplémentaire par bin
# ─────────────────────────────────────────────────────────────────────────────

def _sample_augmented(csv_aug, label_col="FaceOcclusion", gender_col="gender", random_seed=42):
    """Charge data_aug.csv et sélectionne un sous-ensemble par bin pour ne pas noyer le signal original."""
    df_aug = pd.read_csv(csv_aug, delimiter=",").dropna()
    df_aug["filename"] = df_aug["filename"].apply(lambda x: x.replace("\\", "/").split("/")[-1])

    bin_targets = {
        (0.20, 0.30): 1_300, (0.30, 0.40): 3_700, (0.40, 0.50): 3_800,
        (0.50, 0.60): 2_997, (0.60, 0.70): 2_500, (0.70, 0.80): 2_000,
        (0.80, 0.90): 1_500, (0.90, 1.01): 1_000,
    }

    rng, frames = np.random.RandomState(random_seed), []
    for (lo, hi), n_target in bin_targets.items():
        mask   = (df_aug[label_col] >= lo) & (df_aug[label_col] < hi)
        subset = df_aug[mask]
        n_take = min(n_target, len(subset))
        if n_take > 0:
            frames.append(subset.sample(n=n_take, random_state=rng, replace=False))
            print(f"    Aug [{lo*100:.0f}–{hi*100:.0f}%) : {n_take:>5,}  (dispo={len(subset):,})")

    df_result = pd.concat(frames, ignore_index=True)
    print(f"  → Total aug : {len(df_result):,} images")
    return df_result[[label_col, gender_col, "filename"]]


# ─────────────────────────────────────────────────────────────────────────────
# build_dataloaders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    csv_train=None, csv_test=None, image_dir=None,
    csv_aug=None, image_dir_aug=None,
    val_size=20_000, batch_size=32, num_workers=0,
    balance_gender=True, gender_col="gender", label_col="FaceOcclusion",
    filename_col="filename", random_seed=42,
) -> dict:
    """Construit les DataLoaders train/train_orig/val/test avec split stratifié."""
    csv_train     = csv_train     or CSV_TRAIN
    csv_test      = csv_test      or CSV_TEST
    image_dir     = image_dir     or IMAGE_DIR
    csv_aug       = csv_aug       or CSV_AUG
    image_dir_aug = image_dir_aug or IMAGE_DIR_AUG

    from sklearn.model_selection import train_test_split

    df_all  = pd.read_csv(csv_train, delimiter=",").dropna().sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_test = pd.read_csv(csv_test,  delimiter=",").dropna()

    df_all["occ_bin"]   = pd.qcut(df_all[label_col], q=10, labels=False, duplicates="drop")
    df_all["strat_key"] = df_all[gender_col].astype(str) + "_" + df_all["occ_bin"].astype(str)

    df_train, df_val = train_test_split(
        df_all, test_size=val_size / len(df_all),
        stratify=df_all["strat_key"], random_state=random_seed,
    )

    df_train_orig = df_train.drop(["occ_bin", "strat_key"], axis=1).reset_index(drop=True)
    df_val        = df_val.drop(  ["occ_bin", "strat_key"], axis=1).reset_index(drop=True)
    df_train_orig["_aug"] = False
    df_val["_aug"]        = False

    if csv_aug and os.path.isfile(csv_aug):
        print(f"\nIntégration dataset supplémentaire :")
        df_aug_sel         = _sample_augmented(csv_aug, label_col=label_col, gender_col=gender_col, random_seed=random_seed)
        df_aug_sel["_aug"] = True
        df_train           = pd.concat([df_train_orig, df_aug_sel], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    else:
        df_train = df_train_orig.copy()
        if csv_aug:
            print(f"  ⚠ CSV aug introuvable : {csv_aug}")

    print(f"\nTrain : {len(df_train):,}  (orig={len(df_train_orig):,} + aug={len(df_train)-len(df_train_orig):,})  |  Val : {len(df_val):,}  |  Test : {len(df_test):,}")

    train_dataset      = FaceOcclusionDataset(df_train,      image_dir, image_dir_aug=image_dir_aug, mode="train")
    train_dataset_orig = FaceOcclusionDataset(df_train_orig, image_dir, mode="train")
    val_dataset        = FaceOcclusionDataset(df_val,        image_dir, mode="val")
    test_dataset       = FaceOcclusionDataset(df_test,       image_dir, mode="test")

    train_sampler = train_sampler_orig = None
    shuffle_train = True
    if balance_gender:
        train_sampler      = make_combined_sampler(df_train,      gender_col=gender_col, occ_col=label_col)
        train_sampler_orig = make_combined_sampler(df_train_orig, gender_col=gender_col, occ_col=label_col)
        shuffle_train      = False

    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    return {
        "train":         DataLoader(train_dataset,      batch_size=batch_size, sampler=train_sampler,      shuffle=shuffle_train, **common),
        "train_orig":    DataLoader(train_dataset_orig, batch_size=batch_size, sampler=train_sampler_orig, shuffle=shuffle_train, **common),
        "val":           DataLoader(val_dataset,        batch_size=batch_size, shuffle=False, **common),
        "test":          DataLoader(test_dataset,       batch_size=batch_size, shuffle=False, **common),
        "df_train":      df_train,
        "df_train_orig": df_train_orig,
        "df_val":        df_val,
        "df_test":       df_test,
    }
