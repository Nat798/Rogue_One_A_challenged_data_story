"""
paths.py — source unique de vérité pour tous les chemins du projet.

Structure du projet :
  projet/
  ├── paths.py
  ├── DATA_GENE/          ← step1, step2, step3
  ├── DATA/
  │   ├── occlusions/     ← train.csv, test_students.csv, data_aug.csv, clean_faces.csv
  │   ├── DATA_AUG/       ← images augmentées générées par step2
  │   ├── crops/
  │   │   └── Crop_224_5fp_100K/   ← images du dataset principal
  │   └── CelebAMask-HQ/  ← dataset source pour la génération
  │       ├── CelebA-HQ-img/
  │       ├── CelebAMask-HQ-mask-anno/
  │       └── CelebAMask-HQ-attribute-anno.txt
  │   └── coco_object/    ← occludants COCO
  │   └── dtd/            ← occludants DTD
  └── TRAIN/              ← model_dino.py, dataset.py, train.py, run_train.py, ...
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Racine du projet — dossier contenant paths.py
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Dossiers principaux
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(ROOT, "DATA")
DATA_GENE_DIR = os.path.join(ROOT, "DATA_GENE")
TRAIN_DIR     = os.path.join(ROOT, "TRAIN")

# ─────────────────────────────────────────────────────────────────────────────
# CSVs — tous dans DATA/occlusions/
# ─────────────────────────────────────────────────────────────────────────────
OCCLUSIONS_DIR   = os.path.join(DATA_DIR, "occlusion_datasets")
CSV_TRAIN        = os.path.join(OCCLUSIONS_DIR, "train.csv")
CSV_TEST         = os.path.join(OCCLUSIONS_DIR, "test_students.csv")
CSV_AUG          = os.path.join(OCCLUSIONS_DIR, "data_aug.csv")
CSV_CLEAN_FACES  = os.path.join(OCCLUSIONS_DIR, "clean_faces.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DIR     = os.path.join(DATA_DIR, "crops", "Crop_224_5fp_100K")
IMAGE_DIR_AUG = os.path.join(DATA_DIR, "DATA_AUG")

# ─────────────────────────────────────────────────────────────────────────────
# CelebAMask-HQ
# ─────────────────────────────────────────────────────────────────────────────
CELEBAMASK_ROOT = os.path.join(DATA_DIR, "CelebAMask-HQ")
CELEB_IMG_DIR   = os.path.join(CELEBAMASK_ROOT, "CelebA-HQ-img")
CELEB_MASK_DIR  = os.path.join(CELEBAMASK_ROOT, "CelebAMask-HQ-mask-anno")
CELEB_ATTR_FILE = os.path.join(CELEBAMASK_ROOT, "CelebAMask-HQ-attribute-anno.txt")

# ─────────────────────────────────────────────────────────────────────────────
# Occludants (COCO + DTD)
# ─────────────────────────────────────────────────────────────────────────────
COCO_IMG_DIR  = os.path.join(DATA_DIR, "coco_object", "object_image_sr")
COCO_MASK_DIR = os.path.join(DATA_DIR, "coco_object", "object_mask_x4")
DTD_IMG_DIR   = os.path.join(DATA_DIR, "dtd", "images")

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints et runs — dans TRAIN/
# ─────────────────────────────────────────────────────────────────────────────
RUNS_DIR           = os.path.join(TRAIN_DIR, "runs")
CHECKPOINT_PHASE1  = os.path.join(RUNS_DIR, "vitb_explora_aug",  "best_model.pth")
CHECKPOINT_PHASE2  = os.path.join(RUNS_DIR, "vitb_phase2_orig",  "best_model.pth")
SAVE_DIR_PHASE1    = os.path.join(RUNS_DIR, "vitb_explora_aug")
SAVE_DIR_PHASE2    = os.path.join(RUNS_DIR, "vitb_phase2_orig")

# ─────────────────────────────────────────────────────────────────────────────
# Sorties analyse et soumission
# ─────────────────────────────────────────────────────────────────────────────
ANALYSE_DIR    = os.path.join(TRAIN_DIR, "analyse_erreurs")
SUBMISSION_CSV = os.path.join(TRAIN_DIR, "submission_tta.csv")
