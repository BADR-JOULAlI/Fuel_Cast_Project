# Exécution sur Lightning AI

Lightning AI Studio est adapté si ta machine locale manque de CPU/RAM. Le projet
est prêt pour un usage simple dans un Studio.

## Étapes dans Lightning AI Studio

1. Créer un nouveau Studio sur Lightning AI.
2. Cloner le dépôt GitHub du projet.
3. Installer les dépendances:

```bash
pip install -r requirements.txt
```

4. Lancer l'entraînement:

```bash
PYTHONPATH=src python scripts/train_model.py --model random_forest --output-dir artifacts
```

Si le calcul reste trop long, commence avec un échantillon:

```bash
PYTHONPATH=src python scripts/train_model.py --model random_forest --sample-size 30000 --output-dir artifacts
```

5. Pousser le modèle vers Hugging Face.

Définir un token Hugging Face dans l'environnement:

```bash
export HF_TOKEN=hf_xxx
```

Puis:

```bash
PYTHONPATH=src python scripts/train_model.py \
  --model random_forest \
  --output-dir artifacts \
  --push-to-hub \
  --hf-repo-id username/fuelcast-model
```

Le projet fullstack sera séparé et utilisera le repo Hugging Face créé ici.
