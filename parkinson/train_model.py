"""Script d'entraînement du modèle MLP pour détection de Parkinson.

Usage: exécuter ce fichier directement pour entraîner et sauvegarder le scaler et le modèle
dans le dossier `model_result/`.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --------------------------
# 1️ Charger les données
# --------------------------
dataset_path = "dataset/parkinsons.data"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset non trouvé: {dataset_path}")

data = pd.read_csv(dataset_path)

# Supprimer la colonne non feature si nécessaire
if 'name' in data.columns:
    data = data.drop(columns=['name'])

# Features et target
X = data.drop(columns=['status']).values
y = data['status'].values

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------
# 2️ Normalisation
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarder le scaler pour la prédiction
os.makedirs("model_result", exist_ok=True)

# Sauvegarder le scaler pour la prédiction
scaler_path = os.path.join("model_result", "scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler sauvegardé -> {scaler_path}")

# --------------------------
# 3️ SMOTE pour équilibrer les classes
# --------------------------
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"Classes après SMOTE : {pd.Series(y_train_balanced).value_counts().to_dict()}")



# --------------------------
# 4️ GridSearchCV MLP
# --------------------------
param_grid = {
    'hidden_layer_sizes': [(32,6), (64,12), (50,25), (100,50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [0.0001, 0.001],
    'max_iter': [500]
}

mlp = MLPClassifier(random_state=42, early_stopping=True)

grid = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=2
)

print(" Recherche des meilleurs paramètres avec GridSearchCV...")
try:
    # Utiliser les données équilibrées par SMOTE pour l'entraînement
    grid.fit(X_train_balanced, y_train_balanced)
except Exception as e:
    print("Erreur pendant GridSearchCV:", e)
    raise

print("\n Meilleurs paramètres :")
print(grid.best_params_)
print(f"Score CV : {grid.best_score_:.4f}")

# --------------------------
# 5️ Meilleur modèle
# --------------------------
best_mlp = grid.best_estimator_
y_pred = best_mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n Accuracy sur test :", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------
# 6️ Matrice de confusion
# --------------------------
cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :")
print(cm)

# --------------------------
# 7️ Sauvegarder le modèle
# --------------------------
model_path = os.path.join("model_result", "mlp_model.joblib")
joblib.dump(best_mlp, model_path)
print(f" Modèle sauvegardé -> {model_path}")


if __name__ == '__main__':
    # Permet d'exécuter le script directement
    pass
