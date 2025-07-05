# PyroBlur - YOLOv11 Face and License Plate Detection

Un projet minimal pour fine-tuner YOLOv11 afin de détecter les visages et plaques d'immatriculation dans une approche unifiée.

## 🚀 Démarrage rapide

### Avec Docker (Recommandé)

```bash
# Construire l'image
docker-compose build

# Lancer le pipeline complet
docker-compose run pyro-blur python main.py full --max-samples 1000 --epochs 50

# Ou étape par étape
docker-compose run pyro-blur python main.py download --max-samples 1000
docker-compose run pyro-blur python main.py train --epochs 50 --batch 16
```

### Installation locale

```bash
# Installation des dépendances
pip install -r requirements.txt

# Ou avec pyproject.toml
pip install -e .
```

## 📁 Structure du projet

```
pyro-blur/
├── data/                    # Datasets (généré automatiquement)
├── models/                  # Modèles entraînés
├── scripts/                 # Scripts d'entraînement et benchmark
├── pyro_blur/              # Package Python principal
├── docker/                 # Configuration Docker
├── main.py                 # Point d'entrée CLI
├── data.yaml               # Configuration dataset YOLO
└── requirements.txt        # Dépendances
```

## 🎯 Utilisation

### 1. Télécharger le dataset

```bash
python main.py download --max-samples 1000
```

### 2. Entraîner le modèle

```bash
python main.py train --epochs 100 --batch 16 --device auto
```

### 3. Benchmarker le modèle

```bash
python main.py benchmark --model models/yolov11l_*.pt --num-samples 100
```

### 4. Faire de l'inférence

```bash
python main.py inference --model models/yolov11l_*.pt --source data/test/images
```

### 5. Pipeline complet

```bash
python main.py full --max-samples 1000 --epochs 50 --batch 16
```

## 🐳 Docker

### Services disponibles

```bash
# Service principal
docker-compose up pyro-blur

# Jupyter notebook
docker-compose up jupyter
```

### Commandes Docker

```bash
# Pipeline complet
docker-compose run pyro-blur ./docker/docker-init.sh full

# Commandes individuelles
docker-compose run pyro-blur ./docker/docker-init.sh download
docker-compose run pyro-blur ./docker/docker-init.sh train
docker-compose run pyro-blur ./docker/docker-init.sh benchmark
```

## 📊 Classes détectées

- **Classe 0**: `face` (visages)
- **Classe 1**: `license_plate` (plaques d'immatriculation)

## 🔧 Configuration

### Paramètres d'entraînement

Le modèle utilise YOLOv11l avec les paramètres suivants :

- **Taille d'image**: 640x640
- **Batch size**: 16 (ajustable)
- **Epochs**: 100 (ajustable)
- **Optimiseur**: AdamW
- **Learning rate**: 0.01

### Datasets

Le projet utilise des données synthétiques pour les tests. Pour des données réelles :

1. **Visages** : WIDER FACE dataset (converti au format YOLO)
2. **Plaques** : Roboflow License Plate Recognition dataset

## 📈 Résultats

Les résultats d'entraînement et de benchmark sont sauvegardés dans :

- `results/` : Rapports de benchmark
- `runs/` : Logs d'entraînement YOLO
- `models/` : Modèles entraînés

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👤 Auteur

**Thomas**
- Email: thomas@grammatico.me
- GitHub: [@ThomGram](https://github.com/ThomGram)

## 🙏 Remerciements

- [Ultralytics](https://github.com/ultralytics/ultralytics) pour YOLOv11
- [Roboflow](https://roboflow.com/) pour les datasets
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) pour le dataset de visages