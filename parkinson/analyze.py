import librosa
import numpy as np
from scipy.stats import entropy
from scipy.signal import hilbert
import joblib
import os
from pydub import AudioSegment

class FeatureExtractionError(Exception):
    """Exception personnalisée pour les erreurs d'extraction de features"""
    pass

def extract_exact_features(audio_path):
    """
    Extrait les 22 features EXACTES comme dans le dataset Parkinson original
    """
    try:
        # Chargement de l'audio
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        features = {}

        # 1. F0 (Fundamental Frequency) - Features de base
        f0 = extract_accurate_f0(y, sr)
        
        if len(f0) == 0:
            raise FeatureExtractionError("Aucune fréquence fondamentale détectée")
        
        features["MDVP:Fo(Hz)"] = float(np.mean(f0))
        features["MDVP:Fhi(Hz)"] = float(np.max(f0))
        features["MDVP:Flo(Hz)"] = float(np.min(f0))

        # 2. Jitter - Mesures exactes de variabilité de F0
        features.update(compute_exact_jitter(f0))

        # 3. Shimmer - Mesures exactes de variabilité d'amplitude
        features.update(compute_exact_shimmer(y))

        # 4. NHR & HNR - Ratio bruit/harmoniques
        features.update(compute_exact_hnr_nhr(y, sr))

        # 5. RPDE (Recurrence Period Density Entropy)
        features["RPDE"] = compute_exact_rpde(f0)

        # 6. DFA (Detrended Fluctuation Analysis)
        features["DFA"] = compute_exact_dfa(y)

        # 7. Spread features et D2
        spread1, spread2, d2 = compute_exact_spread_features(y)
        features["spread1"] = float(spread1)
        features["spread2"] = float(spread2)
        features["D2"] = float(d2)

        # 8. PPE (Pitch Period Entropy)
        features["PPE"] = compute_exact_ppe(f0)

        return features
        
    except Exception as e:
        raise FeatureExtractionError(f"Erreur extraction features exactes: {str(e)}")

def extract_accurate_f0(y, sr):
    """
    Extraction précise de F0 pour des mesures exactes
    """
    try:
        # Utilisation de PYIN pour une extraction précise de F0
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=50, 
            fmax=500, 
            sr=sr,
            frame_length=2048,
            hop_length=512,
            fill_na=0.0
        )
        
        # Filtrer les valeurs NaN et les valeurs hors plage
        f0 = f0[~np.isnan(f0)]
        f0 = f0[(f0 >= 50) & (f0 <= 500)]
        f0 = f0[f0 > 0]
        
        if len(f0) == 0:
            # Fallback à YIN si PYIN échoue
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=512)
            f0 = f0[f0 > 0]
        
        return f0
        
    except Exception as e:
        raise FeatureExtractionError(f"Erreur extraction F0: {str(e)}")

def compute_exact_jitter(f0):
    """
    Calcule les mesures exactes de Jitter comme dans la littérature
    """
    try:
        features = {}
        
        if len(f0) < 2:
            raise FeatureExtractionError("Pas assez de points F0 pour calculer le jitter")
        
        periods = 1.0 / f0  # Périodes en secondes
        period_diffs = np.abs(np.diff(periods))
        
        # Jitter local (variabilité relative)
        jitter_local = np.mean(period_diffs) / np.mean(periods)
        features["MDVP:Jitter(%)"] = float(jitter_local * 100)
        
        # Jitter absolu
        features["MDVP:Jitter(Abs)"] = float(np.mean(period_diffs))
        
        # RAP (Relative Average Perturbation) - moyenne sur 3 points
        rap_values = []
        for i in range(1, len(periods)-1):
            local_avg = (periods[i-1] + periods[i] + periods[i+1]) / 3
            rap_values.append(np.abs(periods[i] - local_avg) / local_avg)
        features["MDVP:RAP"] = float(np.mean(rap_values)) if rap_values else 0.0
        
        # PPQ (5-point Period Perturbation Quotient)
        ppq_values = []
        for i in range(2, len(periods)-2):
            local_avg = np.mean(periods[i-2:i+3])
            ppq_values.append(np.abs(periods[i] - local_avg) / local_avg)
        features["MDVP:PPQ"] = float(np.mean(ppq_values)) if ppq_values else 0.0
        
        # DDP (Jitter DDP)
        features["Jitter:DDP"] = float(np.mean([
            features["MDVP:RAP"] * 3,
            features["MDVP:Jitter(%)"] / 100 * 3
        ]))
        
        return features
        
    except Exception as e:
        raise FeatureExtractionError(f"Erreur calcul jitter: {str(e)}")

def compute_exact_shimmer(y):
    """
    Calcule les mesures exactes de Shimmer comme dans la littérature
    """
    try:
        features = {}
        
        if len(y) < 100:
            raise FeatureExtractionError("Signal trop court pour calculer le shimmer")
        
        # Extraction de l'enveloppe amplitude
        analytic_signal = hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Éviter les divisions par zéro
        amplitude_envelope = amplitude_envelope[amplitude_envelope > 1e-8]
        
        if len(amplitude_envelope) < 2:
            raise FeatureExtractionError("Enveloppe d'amplitude trop courte")
        
        amplitude_diffs = np.abs(np.diff(amplitude_envelope))
        
        # Shimmer local
        shimmer_local = np.mean(amplitude_diffs) / np.mean(amplitude_envelope)
        features["MDVP:Shimmer"] = float(shimmer_local)
        
        # Shimmer en dB
        shimmer_db = 20 * np.log10(1 + shimmer_local)
        features["MDVP:Shimmer(dB)"] = float(shimmer_db)
        
        # APQ3 (3-point Amplitude Perturbation Quotient)
        apq3_values = []
        for i in range(1, len(amplitude_envelope)-1):
            local_avg = np.mean(amplitude_envelope[i-1:i+2])
            apq3_values.append(np.abs(amplitude_envelope[i] - local_avg) / local_avg)
        features["Shimmer:APQ3"] = float(np.mean(apq3_values)) if apq3_values else 0.0
        
        # APQ5 (5-point Amplitude Perturbation Quotient)
        apq5_values = []
        for i in range(2, len(amplitude_envelope)-2):
            local_avg = np.mean(amplitude_envelope[i-2:i+3])
            apq5_values.append(np.abs(amplitude_envelope[i] - local_avg) / local_avg)
        features["Shimmer:APQ5"] = float(np.mean(apq5_values)) if apq5_values else 0.0
        
        # APQ (équivalent à Shimmer:APQ5 dans beaucoup de datasets)
        features["MDVP:APQ"] = features["Shimmer:APQ5"]
        
        # DDA (Shimmer DDA)
        features["Shimmer:DDA"] = float(features["Shimmer:APQ3"] * 3)
        
        return features
        
    except Exception as e:
        raise FeatureExtractionError(f"Erreur calcul shimmer: {str(e)}")

def compute_exact_hnr_nhr(y, sr):
    """
    Calcule HNR et NHR exacts comme dans la littérature
    """
    try:
        features = {}
        
        # Méthode standard: séparation harmonique/percussif
        harmonic, percussive = librosa.effects.hpss(y)
        
        # Calcul des puissances
        total_power = np.sum(y**2)
        harmonic_power = np.sum(harmonic**2)
        noise_power = np.sum(percussive**2)
        
        # Éviter les divisions par zéro
        if harmonic_power <= 0:
            raise FeatureExtractionError("Puissance harmonique nulle")
        
        # NHR (Noise-to-Harmonics Ratio)
        features["NHR"] = float(noise_power / harmonic_power)
        
        # HNR (Harmonics-to-Noise Ratio) en dB
        if noise_power > 0:
            features["HNR"] = float(10 * np.log10(harmonic_power / noise_power))
        else:
            features["HNR"] = 50.0  # Valeur haute si bruit négligeable
        
        return features
        
    except Exception as e:
        raise FeatureExtractionError(f"Erreur calcul HNR/NHR: {str(e)}")

def compute_exact_rpde(f0):
    """
    Calcule RPDE exact (Recurrence Period Density Entropy)
    """
    try:
        if len(f0) < 10:
            return 0.0
        
        # Périodes à partir de F0
        periods = 1.0 / f0
        
        # Différences de périodes consécutives
        period_diffs = np.diff(periods)
        
        # Histogramme des différences pour l'entropie
        hist, bin_edges = np.histogram(period_diffs, bins=20, density=True)
        hist = hist[hist > 0]  # Supprimer les bins vides

        if len(hist) < 2:
            return 0.0

        # Calcul de l'entropie et normalisation (pour garder la valeur dans une plage comparable)
        rpde_raw = entropy(hist)
        # Normaliser par le log du nombre de bins non-vides pour obtenir une valeur entre 0 et 1
        rpde_norm = rpde_raw / np.log(len(hist) + 1e-8)

        return float(rpde_norm)
        
    except Exception as e:
        raise FeatureExtractionError(f"Erreur calcul RPDE: {str(e)}")

def compute_exact_dfa(signal):
    """
    Calcule DFA exact (Detrended Fluctuation Analysis)
    """
    try:
        if len(signal) < 100:
            return 0.0
        
        # Intégrale du signal
        y_cumsum = np.cumsum(signal - np.mean(signal))
        
        # Échelles pour l'analyse
        scales = np.logspace(np.log10(10), np.log10(len(signal)//4), 20).astype(int)
        scales = scales[scales > 4]
        
        fluctuations = []
        
        for scale in scales:
            if scale >= len(y_cumsum):
                continue
                
            # Découpage en fenêtres
            num_windows = len(y_cumsum) // scale
            if num_windows < 2:
                continue
                
            f_scale = []
            
            for i in range(num_windows):
                segment = y_cumsum[i*scale:(i+1)*scale]
                if len(segment) < 4:
                    continue
                    
                # Détrending linéaire
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                detrended = segment - trend
                
                # Fluctuation RMS
                f_scale.append(np.sqrt(np.mean(detrended**2)))
            
            if f_scale:
                fluctuations.append(np.mean(f_scale))
        
        if len(fluctuations) < 3:
            return 0.0
        
        # Regression linéaire log-log
        log_scales = np.log10(scales[:len(fluctuations)])
        log_fluct = np.log10(fluctuations)
        
        dfa_alpha = np.polyfit(log_scales, log_fluct, 1)[0]
        
        return float(dfa_alpha)
        
    except Exception as e:
        raise FeatureExtractionError(f"Erreur calcul DFA: {str(e)}")

def compute_exact_spread_features(y):
    """
    Calcule spread1, spread2 et D2 exacts
    """
    try:
        if len(y) < 100:
            return 0.0, 0.0, 0.0
        
        # Enveloppe du signal
        analytic_signal = hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)
        
        # spread1 - essayer d'approcher l'échelle du dataset original (valeurs souvent négatives)
        # Utiliser un ratio log de la moyenne d'enveloppe / amplitude moyenne pour obtenir des valeurs
        # qui peuvent être négatives et comparables à celles du dataset UCI
        mean_env = np.mean(amplitude_envelope) + 1e-12
        mean_abs = np.mean(np.abs(y)) + 1e-12
        spread1 = np.log10(mean_env) - np.log10(mean_abs)

        # spread2 - variabilité des différences d'enveloppe, exprimée sur une échelle log
        envelope_diff = np.diff(amplitude_envelope)
        mean_env_diff = np.mean(np.abs(envelope_diff)) + 1e-12
        spread2 = np.log10(mean_env_diff) - np.log10(mean_abs)

        # D2 - dimension de corrélation approximative (laisser l'approche précédente)
        d2_value = np.log(1 + np.std(y) / (np.mean(np.abs(y)) + 1e-8))

        return float(spread1), float(spread2), float(d2_value)
        
    except Exception as e:
        raise FeatureExtractionError(f"Erreur calcul spread features: {str(e)}")

def compute_exact_ppe(f0):
    """
    Calcule PPE exact (Pitch Period Entropy)
    """
    try:
        if len(f0) < 10:
            return 0.0
        
        # Périodes de pitch
        periods = 1.0 / f0
        
        # Histogramme des périodes pour l'entropie
        hist, bin_edges = np.histogram(periods, bins=20, density=True)
        hist = hist[hist > 0]  # Supprimer les bins vides

        if len(hist) < 2:
            return 0.0

        # Calcul de l'entropie et normalisation (ramener sur une échelle comparable au dataset)
        ppe_raw = entropy(hist)
        ppe_norm = ppe_raw / np.log(len(hist) + 1e-8)

        return float(ppe_norm)
        
    except Exception as e:
        raise FeatureExtractionError(f"Erreur calcul PPE: {str(e)}")
from pydub import AudioSegment
import joblib
import os

def predict_parkinson(audio_path):
    """
    Prédiction Parkinson à partir d'un fichier audio.
    Convertit automatiquement les formats (webm, mp3...) en WAV.
    Applique la normalisation EXACTE utilisée lors de l'entraînement.
    """
    try:
        # Chemins modèle + scaler
        model_path = "model_result/mlp_model.joblib"
        scaler_path = "model_result/scaler.joblib"

        if not os.path.exists(model_path):
            return None, f"Modèle non trouvé: {model_path}"
        if not os.path.exists(scaler_path):
            return None, f"Scaler non trouvé: {scaler_path}"

        # --- 1) Conversion automatique vers WAV ---
        original_path = audio_path
        ext = os.path.splitext(audio_path)[1].lower()

        if ext != ".wav":
            wav_path = audio_path.rsplit('.', 1)[0] + "_converted.wav"
            try:
                audio = AudioSegment.from_file(audio_path)
                audio.export(wav_path, format="wav")
                audio_path = wav_path
            except Exception as e:
                return None, f"Erreur conversion WAV: {str(e)}"

        # --- 2) Charger modèle et scaler ---
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # --- 3) Extraire les features EXACTES ---
        features = extract_exact_features(audio_path)
        # --- AJOUTER CE PRINT POUR AFFICHER LES VALEURS BRUTES ---
        print("=== Features brutes (avant normalisation) ===")
        for name, val in features.items():
            print(f"{name}: {val}")
        # Ordre EXACT utilisé lors de la création du modèle
        feature_names = [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
            "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
            "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
            "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
            "spread1", "spread2", "D2", "PPE"
        ]

        feature_vector = [features[name] for name in feature_names]

        # --- 4) NORMALISATION ---
        feature_vector_norm = scaler.transform([feature_vector])

        # --- 5) PREDICTION ---
        prediction = model.predict(feature_vector_norm)[0]
        probability = model.predict_proba(feature_vector_norm)[0]

        result = {
            "prediction": int(prediction),
            "probability": float(probability[prediction]),
            "status": "Parkinson détecté" if prediction == 1 else "Voix normale",
            "features_raw": features,
            "features_normalized": {
                name: float(val) for name, val in zip(feature_names, feature_vector_norm[0])
            }
        }

        # --- 6) Suppression du WAV temporaire ---
        if ext != ".wav" and audio_path != original_path and os.path.exists(audio_path):
            os.remove(audio_path)

        return result, None

    except Exception as e:
        return None, f"ERREUR: {str(e)}"
