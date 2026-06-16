# step2_generate_occlusion_dataset.py
# Génère un dataset d'occlusion synthétique depuis les visages propres de step1.
# Pour chaque bin d'occlusion [20%-100%], applique des occludants COCO/DTD
# sur les visages CelebAMask-HQ et calcule le taux d'occlusion géométrique.
# Produit : DATA/DATA_AUG/ (images) + DATA/occlusions/data_aug.csv

import os
import sys
import glob
import random
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import (
    CELEB_IMG_DIR, CELEB_MASK_DIR,
    COCO_IMG_DIR, COCO_MASK_DIR, DTD_IMG_DIR,
    CSV_CLEAN_FACES, CSV_AUG, IMAGE_DIR_AUG,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE             = 224
N_PER_BIN_PER_GENDER = 4000
BIN_TOLERANCE        = 0.08
MAX_ATTEMPTS_PER_IMAGE = 3

BINS = [
    (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60),
    (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00),
]

FACE_CLASSES = [
    "skin", "nose", "eye_g", "l_eye", "r_eye",
    "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip", "l_lip",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers masques CelebAMask-HQ
# ─────────────────────────────────────────────────────────────────────────────
def _subfolder_for(img_id: int) -> str:
    return str(img_id // 2000)

def _mask_prefix(img_id: int) -> str:
    return f"{img_id:05d}"

def load_face_mask(img_id: int, target_size: int = IMG_SIZE) -> np.ndarray:
    subfolder = _subfolder_for(img_id)
    prefix    = _mask_prefix(img_id)
    folder    = os.path.join(CELEB_MASK_DIR, subfolder)
    combined  = np.zeros((512, 512), dtype=np.uint8)
    for cls in FACE_CLASSES:
        mask_path = os.path.join(folder, f"{prefix}_{cls}.png")
        if os.path.isfile(mask_path):
            try:
                m = np.array(Image.open(mask_path).convert("L"))
                combined = np.maximum(combined, m)
            except Exception:
                pass
    mask_img = Image.fromarray(combined).resize((target_size, target_size), Image.NEAREST)
    return np.array(mask_img) > 0

# ─────────────────────────────────────────────────────────────────────────────
# Chargement des occludants
# ─────────────────────────────────────────────────────────────────────────────
def load_occluder_list() -> list:
    occluders = []
    if os.path.isdir(COCO_IMG_DIR):
        coco_imgs = (
            glob.glob(os.path.join(COCO_IMG_DIR, "*.jpg")) +
            glob.glob(os.path.join(COCO_IMG_DIR, "*.jpeg")) +
            glob.glob(os.path.join(COCO_IMG_DIR, "*.png"))
        )
        for img_path in coco_imgs:
            base      = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(COCO_MASK_DIR, f"{base}.png")
            if not os.path.isfile(mask_path):
                candidates = glob.glob(os.path.join(COCO_MASK_DIR, f"{base}*"))
                mask_path  = candidates[0] if candidates else None
            occluders.append({"type": "coco", "img_path": img_path, "mask_path": mask_path})
        print(f"  COCO : {len(coco_imgs)}")
    if os.path.isdir(DTD_IMG_DIR):
        dtd_imgs = (
            glob.glob(os.path.join(DTD_IMG_DIR, "**", "*.jpg"), recursive=True) +
            glob.glob(os.path.join(DTD_IMG_DIR, "**", "*.png"), recursive=True)
        )
        for img_path in dtd_imgs:
            occluders.append({"type": "dtd", "img_path": img_path, "mask_path": None})
        print(f"  DTD  : {len(dtd_imgs)}")
    print(f"  Total : {len(occluders)}")
    return occluders

# ─────────────────────────────────────────────────────────────────────────────
# Préchargement en RAM
# ─────────────────────────────────────────────────────────────────────────────
def preload_faces(df_clean: pd.DataFrame, img_dir: str) -> dict:
    print(f"\n── Préchargement visages en RAM ─────────────────────────────────────")
    cache = {}
    for img_id in tqdm(df_clean["img_id"].tolist(), desc="  Visages+masques", unit="img"):
        img_id   = int(img_id)
        src_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.isfile(src_path):
            continue
        try:
            src_img   = np.array(Image.open(src_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))
            face_mask = load_face_mask(img_id, target_size=IMG_SIZE)
            if face_mask.sum() >= 100:
                cache[img_id] = {"img": src_img, "mask": face_mask}
        except Exception:
            pass
    print(f"  {len(cache)} visages en RAM")
    return cache

def preload_occluders(occluders: list, max_occ: int = 2000) -> list:
    print(f"\n── Préchargement occludants en RAM ──────────────────────────────────")
    sampled = occluders.copy()
    random.shuffle(sampled)
    sampled = sampled[:max_occ]
    loaded  = []
    for occ in tqdm(sampled, desc="  Occludants", unit="occ"):
        try:
            img = np.array(Image.open(occ["img_path"]).convert("RGB"))
            if occ["mask_path"] and os.path.isfile(occ["mask_path"]):
                mask = np.array(Image.open(occ["mask_path"]).convert("L")) > 127
            else:
                mask = np.ones((img.shape[0], img.shape[1]), dtype=bool)
            if mask.sum() > 0:
                loaded.append({"type": occ["type"], "img": img, "mask": mask})
        except Exception:
            pass
    print(f"  {len(loaded)} occludants en RAM")
    return loaded

# ─────────────────────────────────────────────────────────────────────────────
# Transfert colorimétrique Reinhard RGB
# ─────────────────────────────────────────────────────────────────────────────
def reinhard_color_transfer(source, target, source_mask, target_mask, strength=0.75):
    eps   = 1e-6
    src_f = source.astype(np.float32)
    tgt_f = target.astype(np.float32)
    src_px, tgt_px = src_f[source_mask], tgt_f[target_mask]
    if len(src_px) == 0 or len(tgt_px) == 0:
        return source
    mu_s, std_s = src_px.mean(axis=0), src_px.std(axis=0) + eps
    mu_t, std_t = tgt_px.mean(axis=0), tgt_px.std(axis=0) + eps
    transferred = (src_f - mu_s) / std_s * std_t + mu_t
    out = src_f.copy()
    out[source_mask] = (1.0 - strength) * src_f[source_mask] + strength * transferred[source_mask]
    return np.clip(out, 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Fallback : ellipse aléatoire
# ─────────────────────────────────────────────────────────────────────────────
def generate_random_ellipse_mask(canvas_size, face_mask, target_lo, target_hi):
    face_pixels  = face_mask.sum()
    target_pix   = int(((target_lo + target_hi) / 2.0) * face_pixels)
    ellipse_area = target_pix * 1.3
    a = int(math.sqrt(ellipse_area / math.pi) * random.uniform(0.8, 1.4))
    b = int(math.sqrt(ellipse_area / math.pi) * random.uniform(0.6, 1.2))
    a = max(5, min(a, canvas_size // 2))
    b = max(5, min(b, canvas_size // 2))
    ys, xs = np.where(face_mask)
    if len(ys) == 0:
        cx, cy = canvas_size // 2, canvas_size // 2
    else:
        idx = random.randint(0, len(ys) - 1)
        cx, cy = int(xs[idx]), int(ys[idx])
    color    = tuple(random.randint(30, 220) for _ in range(3))
    occ_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.ellipse(occ_mask, center=(cx, cy), axes=(a, b),
                angle=random.uniform(0, 180), startAngle=0, endAngle=360,
                color=255, thickness=-1)
    occ_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    occ_img[occ_mask > 0] = color
    return occ_img, (occ_mask > 0)

# ─────────────────────────────────────────────────────────────────────────────
# Application d'un occludant
# ─────────────────────────────────────────────────────────────────────────────
def apply_occluder(src_img, face_mask, occluder, target_lo, target_hi, canvas_size=IMG_SIZE):
    H, W        = src_img.shape[:2]
    face_pixels = int(face_mask.sum())
    if face_pixels == 0:
        return None, None

    target_pix   = ((target_lo + target_hi) / 2.0) * face_pixels
    occ_img_raw  = occluder["img"]
    occ_mask_raw = occluder["mask"]
    occ_type     = occluder["type"]

    occ_area_orig = occ_mask_raw.sum()
    if occ_area_orig == 0:
        occ_mask_raw  = np.ones_like(occ_mask_raw)
        occ_area_orig = occ_mask_raw.sum()

    oh, ow       = occ_img_raw.shape[:2]
    scale_factor = math.sqrt((target_pix / 0.6) / occ_area_orig)
    max_scale    = (canvas_size * 1.2) / min(ow, oh)
    min_scale    = 5.0 / min(ow, oh)
    scale_factor = min(max(scale_factor, min_scale), max_scale) * random.uniform(0.8, 1.2)

    new_w = max(4, int(ow * scale_factor))
    new_h = max(4, int(oh * scale_factor))

    occ_img_rs  = cv2.resize(occ_img_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    occ_mask_rs = cv2.resize(occ_mask_raw.astype(np.uint8) * 255, (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST) > 127

    try:
        face_pixels_src = src_img[face_mask]
        if len(face_pixels_src) > 500:
            idx = np.random.choice(len(face_pixels_src), 500, replace=False)
            face_pixels_src = face_pixels_src[idx]
        src_f   = occ_img_rs.astype(np.float32)
        eps     = 1e-6
        mu_s    = src_f[occ_mask_rs].mean(axis=0) if occ_mask_rs.sum() > 0 else np.zeros(3)
        std_s   = src_f[occ_mask_rs].std(axis=0)  + eps if occ_mask_rs.sum() > 0 else np.ones(3)
        mu_t    = face_pixels_src.astype(np.float32).mean(axis=0)
        std_t   = face_pixels_src.astype(np.float32).std(axis=0) + eps
        strength = 0.65
        transferred = (src_f - mu_s) / std_s * std_t + mu_t
        out = src_f.copy()
        out[occ_mask_rs] = (1.0 - strength) * src_f[occ_mask_rs] + strength * transferred[occ_mask_rs]
        occ_img_rs = np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        pass

    ys, xs = np.where(face_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    cx = random.randint(x_min, x_max)
    cy = random.randint(y_min, y_max)
    x0, y0 = cx - new_w // 2, cy - new_h // 2

    result_img    = src_img.copy()
    occ_on_canvas = np.zeros((H, W), dtype=bool)

    ox0 = max(0, -x0); oy0 = max(0, -y0)
    ox1 = min(new_w, W - x0); oy1 = min(new_h, H - y0)
    cx0 = max(0, x0);  cy0 = max(0, y0)
    cx1 = cx0 + (ox1 - ox0); cy1 = cy0 + (oy1 - oy0)

    if ox1 <= ox0 or oy1 <= oy0 or cx1 <= cx0 or cy1 <= cy0:
        return None, None

    patch_mask = occ_mask_rs[oy0:oy1, ox0:ox1]
    patch_img  = occ_img_rs[oy0:oy1, ox0:ox1]
    result_img[cy0:cy1, cx0:cx1][patch_mask] = patch_img[patch_mask]
    occ_on_canvas[cy0:cy1, cx0:cx1][patch_mask] = True

    intersection   = np.logical_and(occ_on_canvas, face_mask).sum()
    occlusion_rate = intersection / face_pixels

    try:
        kernel  = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(occ_on_canvas.astype(np.uint8) * 255, kernel) > 127
        edge_mask = dilated & ~occ_on_canvas
        blurred   = cv2.GaussianBlur(result_img, (3, 3), 0)
        result_img[edge_mask] = blurred[edge_mask]
    except Exception:
        pass

    return result_img, float(occlusion_rate)

# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale de génération
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(IMAGE_DIR_AUG, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_AUG), exist_ok=True)

    if not os.path.isfile(CSV_CLEAN_FACES):
        raise FileNotFoundError(f"clean_faces.csv introuvable : {CSV_CLEAN_FACES}\nLance d'abord step1.")

    df_clean = pd.read_csv(CSV_CLEAN_FACES, dtype={"gender": str})

    def normalize_gender(g) -> str:
        s = str(g).strip().lower()
        if s in ("1", "1.0", "m", "male"):             return "1"
        if s in ("0", "0.0", "f", "female", "-1"):     return "0"
        return "unknown"

    df_clean["gender"] = df_clean["gender"].apply(normalize_gender)
    n_unknown = (df_clean["gender"] == "unknown").sum()
    if n_unknown == len(df_clean):
        idx  = df_clean.sample(frac=1, random_state=42).index
        half = len(idx) // 2
        df_clean.loc[idx[:half], "gender"] = "0"
        df_clean.loc[idx[half:], "gender"] = "1"

    df_female = df_clean[df_clean["gender"] == "0"].reset_index(drop=True)
    df_male   = df_clean[df_clean["gender"] == "1"].reset_index(drop=True)
    if len(df_female) == 0:
        df_female = df_male.copy(); df_female["gender"] = "0"
    if len(df_male) == 0:
        df_male = df_female.copy(); df_male["gender"] = "1"

    face_cache        = preload_faces(df_clean, CELEB_IMG_DIR)
    occluder_list_raw = load_occluder_list()

    if not occluder_list_raw:
        use_fallback = True
        occ_cache = occ_cache_coco = occ_cache_dtd = []
    else:
        use_fallback  = False
        occ_cache     = preload_occluders(occluder_list_raw, max_occ=2000)
        if not occ_cache:
            use_fallback = True
        else:
            occ_cache_coco = [o for o in occ_cache if o["type"] == "coco"]
            occ_cache_dtd  = [o for o in occ_cache if o["type"] == "dtd"]

    print(f"\n── Génération ───────────────────────────────────────────────────────")
    records, global_idx, failed_bins = [], 0, {}

    for bin_lo, bin_hi in BINS:
        bin_label = f"{int(bin_lo*100)}-{int(bin_hi*100)}%"
        print(f"\n  ── Bin {bin_label} ──────────────────────────────────────────────")

        for gender_str, df_gender in [("0", df_female), ("1", df_male)]:
            gender_label = "F" if gender_str == "0" else "M"
            if len(df_gender) == 0:
                continue

            n_generated = 0
            failed      = 0
            df_shuffled = df_gender.sample(frac=1, random_state=random.randint(0, 9999))
            img_pool    = df_shuffled.to_dict("records")
            pool_idx    = 0
            pbar        = tqdm(total=N_PER_BIN_PER_GENDER, desc=f"    Genre {gender_label}", unit="img")

            while n_generated < N_PER_BIN_PER_GENDER:
                if pool_idx >= len(img_pool):
                    pool_idx = 0
                    img_pool = df_gender.sample(frac=1, random_state=random.randint(0, 9999)).to_dict("records")

                row    = img_pool[pool_idx]; pool_idx += 1
                img_id = int(row["img_id"])
                cached = face_cache.get(img_id)
                if cached is None:
                    failed += 1; continue

                src_img, face_mask = cached["img"], cached["mask"]
                best_result = best_rate = None
                best_dist   = float("inf")

                for _ in range(MAX_ATTEMPTS_PER_IMAGE):
                    if use_fallback or not occ_cache:
                        result_img, rate = generate_random_ellipse_mask(IMG_SIZE, face_mask, bin_lo, bin_hi)
                        result_final     = src_img.copy()
                        ell_mask         = (result_img.sum(axis=2) > 0)
                        result_final[ell_mask] = result_img[ell_mask]
                        result_img = result_final
                    else:
                        if occ_cache_coco and occ_cache_dtd:
                            occ = random.choice(occ_cache_coco) if random.random() < 0.5 else random.choice(occ_cache_dtd)
                        elif occ_cache_coco:
                            occ = random.choice(occ_cache_coco)
                        else:
                            occ = random.choice(occ_cache_dtd)
                        result_img, rate = apply_occluder(src_img, face_mask, occ, target_lo=bin_lo, target_hi=bin_hi)

                    if result_img is None or rate is None:
                        continue

                    dist = abs(rate - (bin_lo + bin_hi) / 2.0)
                    if bin_lo - BIN_TOLERANCE <= rate <= bin_hi + BIN_TOLERANCE:
                        if dist < best_dist:
                            best_result, best_rate, best_dist = result_img, rate, dist
                        if bin_lo <= rate <= bin_hi:
                            break
                    elif dist < best_dist:
                        best_result, best_rate, best_dist = result_img, rate, dist

                if best_result is None:
                    failed += 1; continue

                out_filename = f"{global_idx:07d}_g{gender_str}_b{int(bin_lo*100)}.jpg"
                out_path     = os.path.join(IMAGE_DIR_AUG, out_filename)

                try:
                    cv2.imwrite(out_path, cv2.cvtColor(best_result, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                except Exception:
                    failed += 1; continue

                records.append({
                    "filename":      out_filename,
                    "FaceOcclusion": round(best_rate, 6),
                    "gender":        gender_str,
                })

                global_idx  += 1
                n_generated += 1
                pbar.update(1)

                if global_idx % 500 == 0:
                    pd.DataFrame(records).to_csv(CSV_AUG, index=False)

            pbar.close()
            failed_bins[(bin_label, gender_label)] = failed

    df_out = pd.DataFrame(records)
    df_out.to_csv(CSV_AUG, index=False)
    print(f"\n✓ {len(df_out)} images générées → {CSV_AUG}")


def check_data_structure():
    print("\n── Vérification des chemins ─────────────────────────────────────────")
    for path, name in [(CELEB_IMG_DIR, "CelebA-HQ-img"), (CELEB_MASK_DIR, "mask-anno")]:
        print(f"  {'✓' if os.path.isdir(path) else '✗'} {name} → {path}")
    for path, name in [(COCO_IMG_DIR, "coco_object/images"), (DTD_IMG_DIR, "dtd/images")]:
        print(f"  {'✓' if os.path.isdir(path) else '⚠'} {name} → {path}")
    if not os.path.isfile(CSV_CLEAN_FACES):
        print(f"  ✗ clean_faces.csv → {CSV_CLEAN_FACES}  (lancer step1 d'abord)")
        return False
    df = pd.read_csv(CSV_CLEAN_FACES)
    print(f"  ✓ clean_faces.csv → {len(df)} visages propres")
    return True


if __name__ == "__main__":
    if not check_data_structure():
        print("\n⚠ Certains chemins manquants. Le script continue avec les données disponibles...\n")
    main()
