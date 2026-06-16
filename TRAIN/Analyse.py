# Analyse.py
# Analyse complète des erreurs du modèle sur le set de validation.
# Produit 9 figures + statistiques textuelles dans TRAIN/analyse_erreurs/.

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CSV_TRAIN, IMAGE_DIR, ANALYSE_DIR, CHECKPOINT_PHASE2

from model_dino import SegDinoRegressorV2

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT          = CHECKPOINT_PHASE2
OUTPUT_DIR          = ANALYSE_DIR
VAL_SIZE            = 20_000
BATCH_SIZE          = 32
RANDOM_SEED         = 42
SATURATION_THRESHOLD = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Détection couleur / N&B
# ─────────────────────────────────────────────────────────────────────────────
def is_grayscale(image_path, threshold=SATURATION_THRESHOLD):
    try:
        arr = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        max_c   = np.maximum(np.maximum(r, g), b)
        min_c   = np.minimum(np.minimum(r, g), b)
        sat     = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-8), 0)
        return sat.mean() * 255 < threshold
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Dataset validation
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

VAL_TRANSFORM = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=_MEAN, std=_STD)])

class ValDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True); self.image_dir = image_dir
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(os.path.join(self.image_dir, row["filename"])).convert("RGB")
        return VAL_TRANSFORM(img), np.float32(row["FaceOcclusion"]), str(row["gender"]), str(row["filename"])

# ─────────────────────────────────────────────────────────────────────────────
# Chargement modèle
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nChargement checkpoint : {CHECKPOINT}")
ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
state = ckpt.get("state_dict", ckpt)

model = SegDinoRegressorV2(
    use_lora=True, lora_r=32, lora_alpha=64.0, n_unfrozen_blocks=2,
    intermediate_block_idx=5, attn_block_idx=-1, n_patches=196, dropout=0.0,
).to(device)

model.load_state_dict(state, strict=False)
model.eval()
print(f"✓ Epoch {ckpt.get('epoch','?')}  val_score={ckpt.get('val_score',float('nan')):.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction du split validation (même seed que l'entraînement)
# ─────────────────────────────────────────────────────────────────────────────
print("\nReconstruction du split validation...")
df_all = pd.read_csv(CSV_TRAIN, delimiter=",").dropna().sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
df_all["occ_bin"]   = pd.qcut(df_all["FaceOcclusion"], q=10, labels=False, duplicates="drop")
df_all["strat_key"] = df_all["gender"].astype(str) + "_" + df_all["occ_bin"].astype(str)

_, df_val = train_test_split(df_all, test_size=VAL_SIZE / len(df_all),
                              stratify=df_all["strat_key"], random_state=RANDOM_SEED)
df_val = df_val.drop(["occ_bin", "strat_key"], axis=1).reset_index(drop=True)
print(f"  Val : {len(df_val)} images")

val_loader = DataLoader(ValDataset(df_val, IMAGE_DIR), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ─────────────────────────────────────────────────────────────────────────────
# Inférence
# ─────────────────────────────────────────────────────────────────────────────
print("\nInférence sur le set de validation...")
all_preds, all_targets, all_genders, all_filenames = [], [], [], []

with torch.no_grad():
    for X, y, gender, filename in tqdm(val_loader, desc="Val", unit="batch"):
        X = X.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            preds = model(X)
        all_preds.extend(preds.float().cpu().numpy())
        all_targets.extend(y.numpy())
        all_genders.extend(gender)
        all_filenames.extend(filename)

preds   = np.array(all_preds)
targets = np.array(all_targets)
genders = np.array(all_genders)
errors  = np.abs(preds - targets)
sq_err  = (preds - targets) ** 2

# ─────────────────────────────────────────────────────────────────────────────
# Détection couleur / N&B
# ─────────────────────────────────────────────────────────────────────────────
print("\nDétection couleur / N&B...")
is_bw = np.array([is_grayscale(os.path.join(IMAGE_DIR, f)) for f in tqdm(all_filenames, desc="Analyse images")])
print(f"  N&B : {is_bw.sum():,}  Couleur : {(~is_bw).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers communs
# ─────────────────────────────────────────────────────────────────────────────
def is_female(g):
    return str(g).strip().lower() in ("0", "0.0", "f", "female")

mask_f      = np.array([is_female(g) for g in genders])
mask_m      = ~mask_f
occ_bins    = np.arange(0, 1.01, 0.1)
occ_labels  = [f"{int(occ_bins[i]*100)}–{int(occ_bins[i+1]*100)}%" for i in range(len(occ_bins)-1)]
occ_bin_idx = np.digitize(targets, occ_bins[1:-1])

C_F   = "#E67E8A"; C_M   = "#5B9BD5"
C_COL = "#2ECC71"; C_BW  = "#7F8C8D"; C_ALL = "#F39C12"

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — MAE par bin d'occlusion GT
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle("Erreur par taux d'occlusion GT", fontsize=14, fontweight="bold")

mae_by_bin   = [errors[occ_bin_idx == b].mean() if (occ_bin_idx == b).sum() > 0 else 0 for b in range(len(occ_labels))]
count_by_bin = [(occ_bin_idx == b).sum() for b in range(len(occ_labels))]

ax = axes[0]
bars = ax.bar(occ_labels, mae_by_bin, color=C_ALL, edgecolor="white", linewidth=0.5)
for bar, mae in zip(bars, mae_by_bin):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{mae:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Taux d'occlusion GT"); ax.set_ylabel("MAE"); ax.set_title("MAE par bin"); ax.tick_params(axis='x', rotation=45)

ax = axes[1]
ax.bar(occ_labels, count_by_bin, color="#3498DB", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Taux d'occlusion GT"); ax.set_ylabel("Nombre d'exemples"); ax.set_title("Distribution par bin"); ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_erreur_par_occlusion.png"), dpi=150, bbox_inches="tight")
print("✓ fig1")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Erreur par genre
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Erreurs par genre", fontsize=14, fontweight="bold")

mae_f = errors[mask_f].mean(); mae_m = errors[mask_m].mean()
w_f   = 1/30 + targets[mask_f]; w_m = 1/30 + targets[mask_m]
err_f_challenge = (w_f * (preds[mask_f] - targets[mask_f])**2).sum() / w_f.sum()
err_m_challenge = (w_m * (preds[mask_m] - targets[mask_m])**2).sum() / w_m.sum()

for ax_, vals, title, ylabel in [
    (axes[0], [mae_f, mae_m], "MAE par genre", "MAE"),
    (axes[1], [err_f_challenge, err_m_challenge], "Err challenge par genre", "Err pondérée"),
]:
    bars = ax_.bar(["Femmes (F)", "Hommes (M)"], vals, color=[C_F, C_M], edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax_.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02, f"{v:.5f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_.set_ylabel(ylabel); ax_.set_title(title); ax_.set_ylim(0, max(vals) * 1.3)

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_erreur_par_genre.png"), dpi=150, bbox_inches="tight")
print("✓ fig2")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Erreur par occlusion × genre
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle("Erreur par occlusion × genre", fontsize=14, fontweight="bold")

mae_f_by_bin = [errors[occ_bin_idx == b & mask_f].mean() if ((occ_bin_idx == b) & mask_f).sum() > 0 else np.nan for b in range(len(occ_labels))]
mae_m_by_bin = [errors[occ_bin_idx == b & mask_m].mean() if ((occ_bin_idx == b) & mask_m).sum() > 0 else np.nan for b in range(len(occ_labels))]
# Correction : recalcul correct avec &
mae_f_by_bin = [errors[(occ_bin_idx == b) & mask_f].mean() if ((occ_bin_idx == b) & mask_f).sum() > 0 else np.nan for b in range(len(occ_labels))]
mae_m_by_bin = [errors[(occ_bin_idx == b) & mask_m].mean() if ((occ_bin_idx == b) & mask_m).sum() > 0 else np.nan for b in range(len(occ_labels))]
x = np.arange(len(occ_labels)); width = 0.38

ax = axes[0]
ax.bar(x - width/2, mae_f_by_bin, width, label="Femmes", color=C_F, edgecolor="white")
ax.bar(x + width/2, mae_m_by_bin, width, label="Hommes", color=C_M, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(occ_labels, rotation=45); ax.set_ylabel("MAE"); ax.set_title("MAE × genre × bin"); ax.legend()

ax = axes[1]
ax.plot(occ_labels, mae_f_by_bin, "o-", color=C_F, label="Femmes", linewidth=2)
ax.plot(occ_labels, mae_m_by_bin, "s-", color=C_M, label="Hommes", linewidth=2)
ax.fill_between(range(len(occ_labels)), [v if not np.isnan(v) else 0 for v in mae_f_by_bin], [v if not np.isnan(v) else 0 for v in mae_m_by_bin], alpha=0.15, color="#9B59B6")
ax.set_xticks(range(len(occ_labels))); ax.set_xticklabels(occ_labels, rotation=45); ax.set_ylabel("MAE"); ax.set_title("Courbes MAE × genre"); ax.legend()

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_erreur_genre_occlusion.png"), dpi=150, bbox_inches="tight")
print("✓ fig3")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Couleur vs N&B
# ─────────────────────────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle(f"Erreur couleur vs N&B (seuil sat={SATURATION_THRESHOLD})", fontsize=14, fontweight="bold")

mae_col_by_bin = [errors[(occ_bin_idx == b) & ~is_bw].mean() if ((occ_bin_idx == b) & ~is_bw).sum() > 0 else np.nan for b in range(len(occ_labels))]
mae_bw_by_bin  = [errors[(occ_bin_idx == b) &  is_bw].mean() if ((occ_bin_idx == b) &  is_bw).sum() > 0 else np.nan for b in range(len(occ_labels))]

ax = axes[0]
ax.bar(x - width/2, mae_col_by_bin, width, label=f"Couleur (n={(~is_bw).sum():,})", color=C_COL, edgecolor="white")
ax.bar(x + width/2, mae_bw_by_bin,  width, label=f"N&B (n={is_bw.sum():,})",         color=C_BW,  edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(occ_labels, rotation=45); ax.set_ylabel("MAE"); ax.set_title("MAE couleur vs N&B"); ax.legend()

ratio = [b/c if (c and not np.isnan(c) and c > 0 and b and not np.isnan(b)) else np.nan for b, c in zip(mae_bw_by_bin, mae_col_by_bin)]
ax = axes[1]
valid = [(i, r) for i, r in enumerate(ratio) if r is not None and not np.isnan(r)]
if valid:
    idxs, vals = zip(*valid)
    ax.bar([occ_labels[i] for i in idxs], vals, color=["#E74C3C" if v > 1 else "#27AE60" for v in vals], edgecolor="white")
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
ax.set_ylabel("Ratio MAE N&B / MAE Couleur"); ax.set_title("Ratio N&B/Couleur"); ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_erreur_couleur_nb.png"), dpi=150, bbox_inches="tight")
print("✓ fig4")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Scatter prédictions vs GT
# ─────────────────────────────────────────────────────────────────────────────
fig5, axes = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle("Prédictions vs GT", fontsize=14, fontweight="bold")
sample_idx = np.random.choice(len(targets), min(3000, len(targets)), replace=False)

ax = axes[0]
ax.scatter(targets[sample_idx][mask_f[sample_idx]], preds[sample_idx][mask_f[sample_idx]], alpha=0.3, s=8, color=C_F, label="Femmes")
ax.scatter(targets[sample_idx][mask_m[sample_idx]], preds[sample_idx][mask_m[sample_idx]], alpha=0.3, s=8, color=C_M, label="Hommes")
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5); ax.set_xlabel("GT"); ax.set_ylabel("Prédiction"); ax.set_title("Scatter par genre"); ax.legend(markerscale=3); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

ax = axes[1]
ax.scatter(targets[sample_idx][~is_bw[sample_idx]], preds[sample_idx][~is_bw[sample_idx]], alpha=0.3, s=8, color=C_COL, label="Couleur")
ax.scatter(targets[sample_idx][is_bw[sample_idx]],  preds[sample_idx][is_bw[sample_idx]],  alpha=0.4, s=8, color=C_BW,  label="N&B")
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5); ax.set_xlabel("GT"); ax.set_ylabel("Prédiction"); ax.set_title("Scatter couleur/N&B"); ax.legend(markerscale=3); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, "fig5_scatter_pred_gt.png"), dpi=150, bbox_inches="tight")
print("✓ fig5")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Heatmap genre × occlusion
# ─────────────────────────────────────────────────────────────────────────────
fig6, axes = plt.subplots(1, 2, figsize=(14, 5))
fig6.suptitle("Heatmap MAE — genre × occlusion", fontsize=14, fontweight="bold")

heatmap_data = np.full((2, len(occ_labels)), np.nan)
count_data   = np.zeros((2, len(occ_labels)))
for b in range(len(occ_labels)):
    mask_b = occ_bin_idx == b
    if (mask_b & mask_f).sum() > 0: heatmap_data[0, b] = errors[mask_b & mask_f].mean(); count_data[0, b] = (mask_b & mask_f).sum()
    if (mask_b & mask_m).sum() > 0: heatmap_data[1, b] = errors[mask_b & mask_m].mean(); count_data[1, b] = (mask_b & mask_m).sum()

for ax_, data, title, cmap in [(axes[0], heatmap_data, "MAE", "YlOrRd"), (axes[1], count_data, "Nombre d'exemples", "Blues")]:
    im = ax_.imshow(data, aspect="auto", cmap=cmap)
    ax_.set_xticks(range(len(occ_labels))); ax_.set_xticklabels(occ_labels, rotation=45, fontsize=8)
    ax_.set_yticks([0, 1]); ax_.set_yticklabels(["Femmes", "Hommes"]); ax_.set_title(title)
    plt.colorbar(im, ax=ax_)
    for i in range(2):
        for j in range(len(occ_labels)):
            if not np.isnan(data[i, j]):
                ax_.text(j, i, f"{data[i,j]:.3f}" if title == "MAE" else f"{int(data[i,j])}", ha="center", va="center", fontsize=7)

plt.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, "fig6_heatmap_genre_occlusion.png"), dpi=150, bbox_inches="tight")
print("✓ fig6")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES 7 & 8 — Distribution par bin × genre (val set + dataset complet)
# ─────────────────────────────────────────────────────────────────────────────
def plot_distribution(axes, count_f_, count_m_, title_suffix, n_f, n_m):
    count_all_ = count_f_ + count_m_
    x_ = np.arange(len(occ_labels)); w_ = 0.38

    ax = axes[0, 0]
    bars_f_ = ax.bar(x_ - w_/2, count_f_, w_, label=f"Femmes (n={n_f:,})", color=C_F, edgecolor="white")
    bars_m_ = ax.bar(x_ + w_/2, count_m_, w_, label=f"Hommes (n={n_m:,})", color=C_M, edgecolor="white")
    for bar in [*bars_f_, *bars_m_]:
        h = bar.get_height()
        if h > 0: ax.text(bar.get_x() + bar.get_width()/2, h + max(count_all_)*0.005, f"{int(h):,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_); ax.set_xticklabels(occ_labels, rotation=45, fontsize=8); ax.legend(); ax.set_title("Barres groupées")

    ax = axes[0, 1]
    ax.bar(x_, count_f_, width=0.6, label="Femmes", color=C_F, edgecolor="white")
    ax.bar(x_, count_m_, width=0.6, bottom=count_f_, label="Hommes", color=C_M, edgecolor="white")
    ax.set_xticks(x_); ax.set_xticklabels(occ_labels, rotation=45, fontsize=8); ax.legend(); ax.set_title("Barres empilées")

    pct_f_ = np.where(count_all_ > 0, 100 * count_f_ / count_all_, 0)
    pct_m_ = np.where(count_all_ > 0, 100 * count_m_ / count_all_, 0)
    ax = axes[1, 0]
    ax.bar(x_, pct_f_, width=0.6, label="Femmes %", color=C_F, edgecolor="white")
    ax.bar(x_, pct_m_, width=0.6, bottom=pct_f_, label="Hommes %", color=C_M, edgecolor="white")
    ax.axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x_); ax.set_xticklabels(occ_labels, rotation=45, fontsize=8); ax.legend(); ax.set_title("Proportions %"); ax.set_ylim(0, 108)

    ax = axes[1, 1]
    ax.bar(x_ - w_/2, np.where(count_f_ > 0, count_f_, 0.5), w_, label="Femmes", color=C_F, edgecolor="white")
    ax.bar(x_ + w_/2, np.where(count_m_ > 0, count_m_, 0.5), w_, label="Hommes", color=C_M, edgecolor="white")
    ax.set_yscale("log"); ax.set_xticks(x_); ax.set_xticklabels(occ_labels, rotation=45, fontsize=8); ax.legend(); ax.set_title("Échelle log")

count_f_val = np.array([(mask_f & (occ_bin_idx == b)).sum() for b in range(len(occ_labels))])
count_m_val = np.array([(mask_m & (occ_bin_idx == b)).sum() for b in range(len(occ_labels))])

fig7, axes7 = plt.subplots(2, 2, figsize=(18, 12))
fig7.suptitle("Distribution val set — genre × occlusion", fontsize=14, fontweight="bold")
plot_distribution(axes7, count_f_val, count_m_val, "val", mask_f.sum(), mask_m.sum())
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig7.savefig(os.path.join(OUTPUT_DIR, "fig7_distribution_genre_occlusion.png"), dpi=150, bbox_inches="tight")
print("✓ fig7")

df_full = pd.read_csv(CSV_TRAIN, delimiter=",").dropna()
full_genders  = df_full["gender"].values
full_targets  = df_full["FaceOcclusion"].values
full_mask_f   = np.array([is_female(g) for g in full_genders])
full_mask_m   = ~full_mask_f
full_occ_bins = np.digitize(full_targets, occ_bins[1:-1])
count_f_full  = np.array([(full_mask_f & (full_occ_bins == b)).sum() for b in range(len(occ_labels))])
count_m_full  = np.array([(full_mask_m & (full_occ_bins == b)).sum() for b in range(len(occ_labels))])

fig8, axes8 = plt.subplots(2, 2, figsize=(18, 12))
fig8.suptitle(f"Distribution dataset complet — {len(df_full):,} images", fontsize=14, fontweight="bold")
plot_distribution(axes8, count_f_full, count_m_full, "complet", full_mask_f.sum(), full_mask_m.sum())
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig8.savefig(os.path.join(OUTPUT_DIR, "fig8_distribution_dataset_complet.png"), dpi=150, bbox_inches="tight")
print("✓ fig8")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 + TABLEAU — Score challenge par palier de GT max
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Score par palier de GT max ────────────────────────────────────────")
paliers, rows_table = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], []

for seuil in paliers:
    mask_seuil = targets <= seuil
    if mask_seuil.sum() == 0: continue
    p_s, t_s, g_s = preds[mask_seuil], targets[mask_seuil], genders[mask_seuil]
    mask_f_s = np.array([is_female(g) for g in g_s]); mask_m_s = ~mask_f_s
    n_f, n_m = mask_f_s.sum(), mask_m_s.sum()
    w_all    = 1/30 + t_s
    err_f    = ((1/30 + t_s[mask_f_s]) * (p_s[mask_f_s] - t_s[mask_f_s])**2).sum() / (1/30 + t_s[mask_f_s]).sum() if n_f > 0 else (w_all * (p_s - t_s)**2).sum() / w_all.sum()
    err_m    = ((1/30 + t_s[mask_m_s]) * (p_s[mask_m_s] - t_s[mask_m_s])**2).sum() / (1/30 + t_s[mask_m_s]).sum() if n_m > 0 else (w_all * (p_s - t_s)**2).sum() / w_all.sum()
    score_   = (err_f + err_m) / 2 + abs(err_f - err_m)
    rows_table.append({"seuil": seuil, "n_total": mask_seuil.sum(), "n_F": n_f, "n_M": n_m, "err_F": err_f, "err_M": err_m, "penalite": abs(err_f - err_m), "score": score_})

df_paliers = pd.DataFrame(rows_table)
for _, row in df_paliers.iterrows():
    print(f"  GT ≤ {row['seuil']*100:3.0f}%  n={int(row['n_total']):>7,}  err_F={row['err_F']:.6f}  err_M={row['err_M']:.6f}  score={row['score']:.6f}")

df_paliers.to_csv(os.path.join(OUTPUT_DIR, "score_par_palier.csv"), index=False)

fig9, axes = plt.subplots(1, 3, figsize=(18, 5))
fig9.suptitle("Score challenge par palier de GT max", fontsize=13, fontweight="bold")
seuils_pct = [r["seuil"] * 100 for r in rows_table]
scores_    = [r["score"]    for r in rows_table]
err_fs_    = [r["err_F"]    for r in rows_table]
err_ms_    = [r["err_M"]    for r in rows_table]
penalties_ = [r["penalite"] for r in rows_table]

axes[0].plot(seuils_pct, scores_, "o-", color="#E74C3C", linewidth=2.5); axes[0].axhline(0.00082, color="#27AE60", linestyle="--", linewidth=2); axes[0].set_xlabel("Palier GT max (%)"); axes[0].set_ylabel("Score"); axes[0].set_title("Score par palier"); axes[0].grid(axis="y", alpha=0.3)
axes[1].plot(seuils_pct, err_fs_, "o-", color=C_F, label="Err_F", linewidth=2.5); axes[1].plot(seuils_pct, err_ms_, "s-", color=C_M, label="Err_M", linewidth=2.5); axes[1].fill_between(seuils_pct, err_fs_, err_ms_, alpha=0.15, color="#9B59B6"); axes[1].set_xlabel("Palier GT max (%)"); axes[1].set_title("Err_F et Err_M"); axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
base_ = [(ef + em) / 2 for ef, em in zip(err_fs_, err_ms_)]
axes[2].bar(seuils_pct, base_, width=6, label="(Err_F+Err_M)/2", color="#3498DB", edgecolor="white")
axes[2].bar(seuils_pct, penalties_, width=6, bottom=base_, label="|Err_F−Err_M|", color="#E74C3C", edgecolor="white")
axes[2].axhline(0.00082, color="#27AE60", linestyle="--", linewidth=2); axes[2].set_xlabel("Palier GT max (%)"); axes[2].set_title("Décomposition score"); axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
fig9.savefig(os.path.join(OUTPUT_DIR, "fig9_score_par_palier.png"), dpi=150, bbox_inches="tight")
print("✓ fig9")

# ─────────────────────────────────────────────────────────────────────────────
# Statistiques textuelles
# ─────────────────────────────────────────────────────────────────────────────
w       = 1/30 + targets
err_f_c = (w[mask_f] * (preds[mask_f]-targets[mask_f])**2).sum() / w[mask_f].sum()
err_m_c = (w[mask_m] * (preds[mask_m]-targets[mask_m])**2).sum() / w[mask_m].sum()
score   = (err_f_c + err_m_c) / 2 + abs(err_f_c - err_m_c)

stats = f"""
{'='*60}
STATISTIQUES — SET DE VALIDATION | Epoch {ckpt.get('epoch','?')}
{'='*60}
Score officiel    : {score:.6f}
Err_F (femmes)    : {err_f_c:.6f}  (n={mask_f.sum():,})
Err_M (hommes)    : {err_m_c:.6f}  (n={mask_m.sum():,})
Pénalité genre    : {abs(err_f_c-err_m_c):.6f}  ({100*abs(err_f_c-err_m_c)/score:.1f}% du score)
MAE global        : {errors.mean():.4f}
RMSE global       : {np.sqrt(sq_err.mean()):.4f}
N&B : {is_bw.sum():,} ({100*is_bw.mean():.1f}%)  |  Couleur : {(~is_bw).sum():,} ({100*(~is_bw).mean():.1f}%)
"""
print(stats)
with open(os.path.join(OUTPUT_DIR, "statistiques.txt"), "w", encoding="utf-8") as f:
    f.write(stats)
print(f"✓ statistiques.txt\n✓ Analyse complète — {OUTPUT_DIR}")
