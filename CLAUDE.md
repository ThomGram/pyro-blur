# CLAUDE.md - PyroBlur Project Documentation

## 📋 Prompt utilisateur original

**Date**: 2025-07-05

Aide moi à créer le projet suivant. Je veux finetune yolov11l pour détecter les visages et les plaques d'imatriculation. Avec plusieurs agents en parallèle, trouve un dataset pour finetune le modèle, ecris un script pour télécharger les images, entrainer yolo et tester et tracker ses performances, commit et push sur git (git@github.com:ThomGram/pyro-blur.git), review le code et mettre à jour CLAUDE.md. Crée le code minimal, le projet doit rester simple. Utilise docker pour tout containeriser. Réflchis, planifie et fais une proposition avant de coder. Ecris ce prompt quelque part pour que je m'en souvienne. Pose des questions. ultrathink

## 🎯 Objectifs du projet

1. **Modèle unifié** : YOLOv11l pour détecter visages ET plaques d'immatriculation
2. **Images 640x640** : Format d'entrée standardisé
3. **GPU training** : Utilisation du GPU pour l'entraînement
4. **Dataset limité** : 1000 images par classe pour commencer
5. **Containerisation** : Docker pour la reproductibilité
6. **Code minimal** : Projet simple et maintenable

## 🏗️ Architecture implémentée

### Structure du projet

```
pyro-blur/
├── data/                    # Datasets (auto-générés)
│   ├── train/images/        # Images d'entraînement
│   ├── train/labels/        # Labels YOLO format
│   ├── val/images/          # Images de validation
│   ├── val/labels/          # Labels de validation
│   └── test/images/         # Images de test
├── models/                  # Modèles entraînés
├── results/                 # Résultats de benchmark
├── scripts/                 # Scripts principaux
│   ├── dataset_downloader.py    # Téléchargement datasets
│   ├── train_yolo11.py          # Entraînement YOLOv11
│   └── benchmark_models.py      # Benchmarking
├── pyro_blur/              # Package Python
├── docker/                 # Configuration Docker
├── main.py                 # Point d'entrée CLI
├── data.yaml               # Configuration YOLO
├── requirements.txt        # Dépendances
└── pyproject.toml          # Configuration projet
```

### Classes détectées

- **Classe 0**: `face` (visages)
- **Classe 1**: `license_plate` (plaques d'immatriculation)

## 🔧 Scripts créés

### 1. `scripts/dataset_downloader.py`

**Fonctionnalités** :
- Téléchargement automatique des datasets
- Génération de données synthétiques pour tests
- Support WIDER FACE et Roboflow datasets
- Conversion au format YOLO
- Limitation à 1000 samples par classe

**Usage** :
```bash
python scripts/dataset_downloader.py --max-samples 1000
```

### 2. `scripts/train_yolo11.py`

**Fonctionnalités** :
- Entraînement YOLOv11l unifié
- Configuration GPU automatique
- Paramètres optimisés pour détection multi-classes
- Sauvegarde automatique des modèles
- Support reprise d'entraînement

**Usage** :
```bash
python scripts/train_yolo11.py --epochs 100 --batch 16 --img-size 640
```

### 3. `scripts/benchmark_models.py`

**Fonctionnalités** :
- Validation sur dataset test
- Mesure vitesse d'inférence (FPS)
- Métriques de performance (mAP, précision, rappel)
- Génération de rapports automatiques
- Visualisations des résultats

**Usage** :
```bash
python scripts/benchmark_models.py --model models/yolov11l_*.pt
```

### 4. `main.py`

**Point d'entrée CLI unifié** :
- Commande `download` : Téléchargement datasets
- Commande `train` : Entraînement modèle
- Commande `benchmark` : Benchmarking
- Commande `inference` : Inférence
- Commande `full` : Pipeline complet

## 🐳 Docker

### Configuration

- **Base image** : `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`
- **GPU support** : NVIDIA Docker runtime
- **Volumes** : Montage des dossiers data/, models/, results/
- **Services** : pyro-blur principal + jupyter notebook

### Commandes Docker

```bash
# Pipeline complet
docker-compose run pyro-blur python main.py full

# Commandes individuelles
docker-compose run pyro-blur ./docker/docker-init.sh download
docker-compose run pyro-blur ./docker/docker-init.sh train
docker-compose run pyro-blur ./docker/docker-init.sh benchmark
```

## 📊 Datasets recherchés

### Visages (WIDER FACE)
- **Taille** : 393k faces, 32k images
- **Format** : Conversion YOLO nécessaire
- **URL** : http://shuoyang1213.me/WIDERFACE/
- **Utilisation** : Classe 0 (face)

### Plaques d'immatriculation (Roboflow)
- **Taille** : 10k+ images
- **Format** : YOLO natif
- **URL** : https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
- **Utilisation** : Classe 1 (license_plate)

## 🎯 Paramètres d'entraînement

### Configuration YOLOv11l

```python
train_params = {
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'device': 'auto',
    'optimizer': 'AdamW',
    'lr0': 0.01,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'patience': 20,
    'save_period': 10
}
```

## 📈 Métriques trackées

### Performance
- **mAP50** : Mean Average Precision à IoU=0.5
- **mAP50-95** : Mean Average Precision IoU=0.5:0.95
- **Precision** : Précision globale
- **Recall** : Rappel global
- **F1-Score** : Score F1 calculé

### Vitesse
- **FPS** : Images par seconde
- **Inference time** : Temps d'inférence moyen
- **GPU utilization** : Utilisation GPU

## 🔄 Workflow complet

### 1. Initialisation
```bash
git clone git@github.com:ThomGram/pyro-blur.git
cd pyro-blur
```

### 2. Avec Docker
```bash
docker-compose build
docker-compose run pyro-blur python main.py full --max-samples 1000 --epochs 50
```

### 3. Local
```bash
pip install -r requirements.txt
python main.py download --max-samples 1000
python main.py train --epochs 50 --batch 16
python main.py benchmark --model models/yolov11l_*.pt
```

## 🚀 Prochaines étapes

### Améliorations possibles

1. **Datasets réels** : Intégrer les vrais datasets WIDER FACE et Roboflow
2. **Augmentation** : Ajouter des techniques d'augmentation de données
3. **Hyperparameter tuning** : Optimisation automatique des hyperparamètres
4. **Multi-GPU** : Support entraînement multi-GPU
5. **Export** : Export vers ONNX, TensorRT pour déploiement
6. **Web interface** : Interface web pour upload/inférence
7. **Monitoring** : Intégration Weights & Biases ou MLflow

### Datasets à intégrer

1. **WIDER FACE** : Dataset complet de visages
2. **Roboflow License Plates** : Dataset complet de plaques
3. **Custom datasets** : Ajout de datasets personnalisés
4. **Data augmentation** : Augmentation intelligente des données

## 🔧 Maintenance

### Linting et tests
```bash
# Linting
python -m flake8 pyro_blur/ scripts/
python -m black pyro_blur/ scripts/

# Tests
python -m pytest tests/
```

### Mise à jour dépendances
```bash
pip list --outdated
pip install --upgrade ultralytics torch torchvision
```

## 📝 Notes techniques

### Choix d'architecture
- **YOLOv11l** : Compromis performance/vitesse optimal
- **Format unifié** : Un seul modèle pour les deux classes
- **640x640** : Taille standard pour YOLOv11
- **PyTorch** : Framework principal avec Ultralytics

### Optimisations
- **Mixed precision** : Entraînement FP16 pour plus de vitesse
- **Batch processing** : Traitement par batch pour efficacité
- **GPU memory** : Gestion optimisée de la mémoire GPU
- **Caching** : Cache des données pour accélération

---

*Document maintenu par Claude Code - Dernière mise à jour : 2025-07-05*