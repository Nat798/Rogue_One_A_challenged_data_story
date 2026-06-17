# DataChallenge2026 — Régression d'occlusion faciale

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Téléchargement des données

Tout se télécharge et se place dans le dossier `DATA/` à la racine du projet.

| Donnée | Lien | Destination |
|---|---|---|
| Données de train (CSV + crops) | https://partage.imt.fr/index.php/s/ntYk27ZFCbeKGqW | `DATA/` |
| Masques d'objets COCO (data aug) | https://drive.google.com/drive/folders/15nZETWlGMdcKY6aHbchRsWkUI42KTNs5 | `DATA/` |
| Textures DTD (data aug) | https://www.robots.ox.ac.uk/~vgg/data/dtd/ | `DATA/` |
| CelebAMask-HQ (visages + segmentation) | https://www.kaggle.com/datasets/ipythonx/celebamaskhq | `DATA/` |

Une fois tout extrait, `DATA/` doit ressembler à :

```
DATA/
├── occlusion_datasets/
│   ├── train.csv
│   ├── test_students.csv
│   ├── data_aug.csv            ← généré par step2 (voir plus bas)
│   └── clean_faces.csv         ← généré par step1 (voir plus bas)
├── DATA_AUG/                   ← images générées par step2
├── crops/
│   └── Crop_224_5fp_100K/      ← images du dataset principal
├── CelebAMask-HQ/
│   ├── CelebA-HQ-img/
│   ├── CelebAMask-HQ-mask-anno/
│   └── CelebAMask-HQ-attribute-anno.txt
├── coco_object/
│   ├── object_image_sr/
│   └── object_mask_x4/
└── dtd/
    └── images/
```

Tous les chemins sont centralisés dans `paths.py` à la racine — si votre arborescence diffère, c'est le seul fichier à modifier.

## 3. Génération du dataset augmenté (optionnel)

Ces 3 scripts sont dans `DATA_GENE/` et se lancent dans l'ordre, depuis n'importe où :

```bash
python DATA_GENE/step1_extract_clean_faces.py
python DATA_GENE/step2_generate_occlusion_dataset.py
python DATA_GENE/step3_verify_dataset.py
```

- **step1** extrait les visages CelebAMask-HQ non occultés → `clean_faces.csv`
- **step2** applique des occludants (COCO/DTD) sur ces visages pour générer `data_aug.csv` + les images dans `DATA_AUG/`
- **step3** vérifie la distribution du dataset généré et produit des graphiques de contrôle

step1 nécessite un modèle déjà entraîné (`TRAIN/runs/vitb_explora_aug/best_model.pth`) pour filtrer les visages — sauter cette étape si vous n'avez pas encore de checkpoint, ou utiliser directement `data_aug.csv` s'il est fourni avec les données de train.

## 4. Entraînement

Tous les scripts d'entraînement sont dans `TRAIN/` et se lancent depuis n'importe où :

```bash
# Phase 1 — entraînement avec dataset augmenté
python TRAIN/run_train.py

# Phase 2 — fine-tuning sur dataset original pur (charge le best_model de la phase 1)
python TRAIN/run_phase2.py
```

Les checkpoints, l'historique et la soumission sont sauvegardés dans `TRAIN/runs/`.

Les hyperparamètres se surchargent via variables d'environnement (ou un fichier `.env` à la racine), par exemple :

```bash
N_EPOCHS=50 BATCH_SIZE=16 python TRAIN/run_train.py
```

## 5. Analyse des erreurs

```bash
python TRAIN/Analyse.py
```

Génère 9 figures + un fichier de statistiques dans `TRAIN/analyse_erreurs/`.

## 6. Inférence finale (TTA + ensemble)

```bash
python TRAIN/INFEROTRON_FINAL.py
```

Produit `TRAIN/submission_tta.csv`. Pour ensembler plusieurs checkpoints, éditer la liste `CHECKPOINTS` en tête du fichier.

## Structure du projet

```
.
├── paths.py              ← chemins centralisés — seul fichier à éditer si l'arborescence change
├── requirements.txt
├── DATA/                  ← données téléchargées (voir section 2)
├── DATA_GENE/             ← scripts de génération du dataset augmenté
└── TRAIN/                 ← modèle, entraînement, analyse, inférence
```
