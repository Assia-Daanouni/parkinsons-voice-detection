from flask import Flask, request, jsonify, render_template
import os
import tempfile
from analyze import predict_parkinson
import joblib
import numpy as np


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    rf_model = joblib.load("model_result/mlp_model.joblib")
    rf_scaler = joblib.load("model_result/scaler.joblib")
    print("Random Forest model loaded")
except Exception as e:
    print(f"Error loading Random Forest: {e}")
    rf_model = None
    rf_scaler = None

@app.route('/')
def index():
    return render_template('index2.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Manual form - Uses Random Forest"""
    result = None
    error = None
    
    if request.method == "POST":
        try:
            if rf_model is None or rf_scaler is None:
                error = "Random Forest model not loaded"
            else:
                features = []
                for i in range(1, 23):
                    value = request.form.get(f"f{i}", "0").strip()
                    features.append(float(value) if value else 0.0)
                
                X = np.array([features])
                X_scaled = rf_scaler.transform(X)
                pred = rf_model.predict(X_scaled)[0]
                proba = rf_model.predict_proba(X_scaled)[0][pred] * 100
                
                if pred == 1:
                    result = {
                        "message": "Parkinson Risk Detected",
                        "details": f"Model Confidence: {proba:.1f}%",
                        "type": "warning",
                        "recommendation": "Consult a neurologist for a comprehensive evaluation."
                    }
                else:
                    result = {
                        "message": "No Risk Detected",
                        "details": f"Model Confidence: {proba:.1f}%",
                        "type": "success",
                        "recommendation": "Continue maintaining good health habits."
                    }
                    
        except Exception as e:
            error = f"Error: {str(e)}"
    
    return render_template("prediction.html", result=result, error=error)

@app.route("/audio")
def audio_page():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze_voice():
    if 'voice' not in request.files:
        return jsonify({'error': 'Aucun fichier audio'}), 400
    
    audio_file = request.files['voice']
    
    if audio_file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    # Sauvegarder temporairement le fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
        audio_file.save(tmp.name)
        temp_path = tmp.name
    
    try:
        # Faire la prédiction
        result, error = predict_parkinson(temp_path)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500
        
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)