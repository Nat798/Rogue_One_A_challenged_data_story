# step3_verify_dataset.py
# Vérifie et visualise le dataset généré par step2.
# Affiche la distribution par bin/genre, calcule le score challenge avec GT=pred,
# et génère une grille visuelle d'exemples.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CSV_AUG, IMAGE_DIR_AUG, DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR           = DATA_DIR
N_PER_BIN_PER_GENDER = 4000

BINS = [
    (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60),
    (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00),
]

# ─────────────────────────────────────────────────────────────────────────────
# Score challenge
# ─────────────────────────────────────────────────────────────────────────────
def _is_female(g) -> bool:
    return str(g).strip().lower() in ("0", "0.0", "f", "female")

def weighted_err(preds: np.ndarray, targets: np.ndarray) -> float:
    w = 1.0 / 30.0 + targets
    return float(np.sum(w * (preds - targets) ** 2) / np.sum(w))

def challenge_score(preds, targets, genders):
    mask_f = np.array([_is_female(g) for g in genders])
    mask_m = ~mask_f
    err_f  = weighted_err(preds[mask_f], targets[mask_f]) if mask_f.sum() > 0 else 0.0
    err_m  = weighted_err(preds[mask_m], targets[mask_m]) if mask_m.sum() > 0 else 0.0
    score  = (err_f + err_m) / 2.0 + abs(err_f - err_m)
    return score, err_f, err_m

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.isfile(CSV_AUG):
        raise FileNotFoundError(f"CSV introuvable : {CSV_AUG}")

    df = pd.read_csv(CSV_AUG)
    print(f"Dataset : {len(df)} images\n")

    bin_labels = [f"{int(lo*100)}-{int(hi*100)}%" for lo, hi in BINS]
    df["bin"]  = pd.cut(
        df["FaceOcclusion"],
        bins   = [lo for lo, _ in BINS] + [1.0],
        labels = bin_labels,
        right  = False,
    )

    print("── Distribution par bin et genre ────────────────────────────────────")
    pivot = df.groupby(["bin", "gender"]).size().unstack(fill_value=0)
    print(pivot.to_string())

    n_f = (df["gender"] == "0").sum()
    n_m = (df["gender"] == "1").sum()
    print(f"\nFemmes : {n_f}  |  Hommes : {n_m}  |  Ratio : {n_f/max(n_m,1):.2f}")

    targets = df["FaceOcclusion"].values
    score, err_f, err_m = challenge_score(targets, targets, df["gender"].values)
    print(f"\n── Score challenge (GT = pred) ───────────────────────────────────────")
    print(f"  Score  : {score:.8f}  (doit être ≈ 0)")
    print(f"  Err_F  : {err_f:.8f} | Err_M  : {err_m:.8f}")

    print(f"\n── Stats FaceOcclusion ──────────────────────────────────────────────")
    print(df["FaceOcclusion"].describe().round(4).to_string())

    # ── Histogramme ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins_edges = np.linspace(0, 1, 41)
    ax.hist(df[df["gender"] == "0"]["FaceOcclusion"].values, bins=bins_edges, alpha=0.6, label="Femmes (0)", color="#E05C8A")
    ax.hist(df[df["gender"] == "1"]["FaceOcclusion"].values, bins=bins_edges, alpha=0.6, label="Hommes (1)", color="#4A90D9")
    ax.set_xlabel("Taux d'occlusion (GT)"); ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des taux d'occlusion"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    counts_f = [((df["gender"] == "0") & (df["bin"] == lbl)).sum() for lbl in bin_labels]
    counts_m = [((df["gender"] == "1") & (df["bin"] == lbl)).sum() for lbl in bin_labels]
    x = np.arange(len(bin_labels)); w = 0.35
    ax.bar(x - w/2, counts_f, w, label="Femmes (0)", color="#E05C8A", alpha=0.8)
    ax.bar(x + w/2, counts_m, w, label="Hommes (1)", color="#4A90D9", alpha=0.8)
    ax.axhline(N_PER_BIN_PER_GENDER, color="red", linestyle="--", alpha=0.5, label=f"Cible {N_PER_BIN_PER_GENDER}")
    ax.set_xticks(x); ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_ylabel("Nombre d'images"); ax.set_title("Répartition par bin"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "distribution_aug.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"\n✓ Histogramme : {plot_path}")

    # ── Grille visuelle ───────────────────────────────────────────────────────
    n_bins_show = min(4, len(BINS))
    fig, axes   = plt.subplots(n_bins_show, 4, figsize=(10, n_bins_show * 2.8))

    for row_idx, (bin_lo, bin_hi) in enumerate(BINS[:n_bins_show]):
        bin_lbl  = bin_labels[row_idx]
        bin_mask = (df["FaceOcclusion"] >= bin_lo) & (df["FaceOcclusion"] < bin_hi)

        for col_idx in range(4):
            ax            = axes[row_idx, col_idx] if n_bins_show > 1 else axes[col_idx]
            gender_target = "0" if col_idx < 2 else "1"
            g_mask        = bin_mask & (df["gender"] == gender_target)
            samples       = df[g_mask]
            if len(samples) == 0:
                ax.axis("off"); continue

            sample   = samples.sample(1).iloc[0]
            img_path = os.path.join(IMAGE_DIR_AUG, sample["filename"])
            try:
                ax.imshow(Image.open(img_path).convert("RGB"))
            except Exception:
                ax.axis("off"); continue

            g_label = "F" if gender_target == "0" else "M"
            ax.set_title(f"{g_label} | {sample['FaceOcclusion']*100:.1f}%", fontsize=8, pad=2,
                         color="#E05C8A" if gender_target == "0" else "#4A90D9")
            ax.axis("off")

        if n_bins_show > 1:
            axes[row_idx, 0].set_ylabel(bin_lbl, fontsize=9, labelpad=5)

    plt.suptitle("Exemples du dataset d'occlusion synthétique", fontsize=11, y=1.01)
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "sample_grid_aug.png")
    plt.savefig(grid_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"✓ Grille visuelle : {grid_path}")
    print(f"\n✓ Vérification terminée — dataset prêt pour l'entraînement")


if __name__ == "__main__":
    main()
