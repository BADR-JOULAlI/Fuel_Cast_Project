# FuelCast Training Project

Projet de prédiction de consommation fuel pour le dataset
`krohnedigital/FuelCast`.

Le notebook principal reste disponible dans `Fuel_Cast_Project.ipynb`. Cette
structure sert uniquement à entraîner le modèle sur Lightning AI et à pousser
les artefacts vers Hugging Face.

## Structure

```text
Fuel_Cast_Project.ipynb        Notebook d'analyse et de modélisation
src/fuelcast/                  Préprocessing réutilisable
scripts/train_model.py         Entraînement + export du modèle
lightning_ai/                  Instructions et script pour Lightning AI
requirements.txt               Dépendances Python
```

## Entraînement local ou Lightning AI

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/train_model.py --model random_forest --output-dir artifacts
```

Pour réduire le coût de calcul pendant les tests:

```bash
PYTHONPATH=src python scripts/train_model.py --model random_forest --sample-size 30000
```

Le script sauvegarde:

- `artifacts/fuelcast_model.joblib`
- `artifacts/metrics.json`
- `artifacts/input_schema.json`
- `artifacts/sample_input.json`
- `artifacts/feature_importances.csv`

## Pousser le modèle vers Hugging Face

```bash
export HF_TOKEN=hf_xxx
PYTHONPATH=src python scripts/train_model.py \
  --model random_forest \
  --output-dir artifacts \
  --push-to-hub \
  --hf-repo-id username/fuelcast-model
```

Le nouveau projet fullstack peut ensuite charger le modèle depuis Hugging Face:

```bash
export HF_MODEL_REPO=username/fuelcast-model
```

Dans cette machine, ce projet séparé est:

```text
C:\Users\anony\Downloads\Fuel_Cast_Fullstack
```

## GitHub puis Lightning AI

Ce dossier est déjà initialisé comme dépôt Git local sur la branche `main`.
Pour le pousser vers GitHub, créer d'abord un repo vide sur GitHub, puis lancer:

```bash
git remote add origin https://github.com/USERNAME/fuelcast-project.git
git push -u origin main
```

Dans Lightning AI Studio, créer un Studio, cloner le repo GitHub, installer
`requirements.txt`, puis lancer les commandes dans `lightning_ai/README.md`.
