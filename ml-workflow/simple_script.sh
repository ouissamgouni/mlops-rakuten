#!/bin/bash

# Variables
MLFLOW_ARTIFACT_PATH="./mlflow_artifacts"
MLFLOW_TRACKING_URI="http://127.0.0.1:8080"
EXPERIMENT_NAME="Rakuten"
MODEL_NAMES=("model_img" "model_text" "final_model")
FILE_NAMES=("model_img.pth" "model_text.sav" "final_model.pth")
GDRIVE_URLS=(
    "https://drive.google.com/uc?id=1jFgRD3DOAej2A7oUqqAD8BjLkmxunril"
    "https://drive.google.com/uc?id=1HZCFCbXdagWwloa_UMLgygu2Gh246yGy"
    "https://drive.google.com/uc?id=1HZCFCbXdagWwloa_UMLgygu2Gh246yGy"
)

# Télécharger les fichiers depuis Google Drive
mkdir -p "$MLFLOW_ARTIFACT_PATH"
for i in "${!GDRIVE_URLS[@]}"; do
    echo "Téléchargement de ${FILE_NAMES[$i]} depuis Google Drive..."
    gdown "${GDRIVE_URLS[$i]}" -O "$MLFLOW_ARTIFACT_PATH/${FILE_NAMES[$i]}"
done

# Script Python pour interagir avec MLflow et gérer l'expérience et les artefacts
python3 - <<EOF
import mlflow
import os

# Variables
MLFLOW_TRACKING_URI = "$MLFLOW_TRACKING_URI"
EXPERIMENT_NAME = "$EXPERIMENT_NAME"
MODEL_NAMES = ["model_img", "model_text", "final_model"]
FILE_NAMES = ["model_img.pth", "model_text.sav", "final_model.pth"]
MLFLOW_ARTIFACT_PATH = "$MLFLOW_ARTIFACT_PATH"

# Configurer l'URI de suivi de MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Vérifier si l'expérience existe
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"Expérience {EXPERIMENT_NAME} créée.")
else:
    print(f"Expérience {EXPERIMENT_NAME} déjà existante.")

# Démarrer un run et enregistrer les artefacts pour chaque modèle
# Démarrer un run et enregistrer les artefacts pour chaque modèle
for i in range(len(MODEL_NAMES)):
    model_name = MODEL_NAMES[i]
    file_name = FILE_NAMES[i]
    
    # Démarrer un run
    with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id):
        print(f"Enregistrement du modèle : {model_name}")
        
        # Ne pas ajouter de slash à la fin du nom du modèle
        artifact_path = model_name  # Pas de '/' ici
        mlflow.log_artifact(os.path.join(MLFLOW_ARTIFACT_PATH, file_name), artifact_path=artifact_path)
        print(f"Modèle {model_name} enregistré avec succès.")

EOF

echo "Tous les modèles et artefacts ont été enregistrés avec succès dans MLflow."
