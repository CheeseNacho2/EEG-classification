import os
import sys
import gc
import numpy as np
import joblib
from tensorflow import keras
from utils import (STRESS_CHANNELS, BANDS, EPOCH_DURATION,
                   OVERLAP, load_and_filter, epoch_signal,
                   extract_band_power, get_feature_names)

# 1. Resolve absolute base path reliably
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
EXPECTED_FEATURES = 28


def load_assets():
    try:
        model  = keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Failed to load model or scaler at {MODEL_PATH}: {e}")

try:
    model, scaler = load_assets()
    print("Model and scaler loaded successfully.")
except RuntimeError as e:
    print(f"Startup error: {e}")
    raise


def pad_features(X, expected_dim=EXPECTED_FEATURES):
    current_dim = X.shape[1]
    if current_dim < expected_dim:
        missing_cols = expected_dim - current_dim
        padding      = np.zeros((X.shape[0], missing_cols))
        return np.hstack((X, padding))
    elif current_dim > expected_dim:
        return X[:, :expected_dim]
    return X


def predict_stress(filepath):
    try:
        raw    = load_and_filter(filepath)
        epochs = epoch_signal(raw)
        if hasattr(raw, 'close'):
            raw.close()
        del raw
        gc.collect()
    except Exception as e:
        raise RuntimeError(f"Failed to load EDF file: {e}")

    if len(epochs) == 0:
        raise ValueError("No epochs could be extracted — recording may be too short.")

    try:
        features = np.array([extract_band_power(epoch) for epoch in epochs])
        features = pad_features(features)
    except Exception as e:
        raise RuntimeError(f"Failed to extract features: {e}")

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("Feature extraction produced invalid values.")

    try:
        features_scaled = scaler.transform(features)
    except Exception as e:
        raise RuntimeError(f"Failed to scale features: {e}")

    try:
        epoch_probs = model.predict(features_scaled, verbose=0).flatten()
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    if len(epoch_probs) == 0:
        raise RuntimeError("Model returned no predictions.")

    mean_prob = float(np.mean(epoch_probs))
    label     = 'Stress' if mean_prob >= 0.5 else 'Calm'

    return {
        'label':       label,
        'probability': round(mean_prob, 4),
        'per_epoch':   epoch_probs.tolist(),
        'n_epochs':    len(epoch_probs)
    }


def retrain(new_filepath, true_label):
    global model, scaler

    if true_label not in [0, 1]:
        raise ValueError(f"Invalid label '{true_label}'. Must be 0 (Calm) or 1 (Stress).")

    try:
        raw    = load_and_filter(new_filepath)
        epochs = epoch_signal(raw)
        if hasattr(raw, 'close'):
            raw.close()
        del raw
        gc.collect()
    except Exception as e:
        raise RuntimeError(f"Failed to load EDF file for retraining: {e}")

    if len(epochs) == 0:
        raise ValueError("No epochs could be extracted — recording may be too short.")

    try:
        features = np.array([extract_band_power(epoch) for epoch in epochs])
        features = pad_features(features)
    except Exception as e:
        raise RuntimeError(f"Failed to extract features for retraining: {e}")

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("Feature extraction produced invalid values.")

    try:
        if hasattr(scaler, 'partial_fit'):
            scaler.partial_fit(features)
        else:
            scaler.fit(features)
        features_scaled = scaler.transform(features)
    except Exception as e:
        raise RuntimeError(f"Failed to scale features for retraining: {e}")

    labels = np.full(len(features), true_label)

    try:
        model.fit(
            features_scaled, labels,
            epochs=10,
            batch_size=32,
            verbose=1
        )
    except Exception as e:
        raise RuntimeError(f"Model retraining failed: {e}")

    # Safely save and release resource handles
    try:
        temp_model_path = MODEL_PATH + ".tmp"
        model.save(temp_model_path)
        joblib.dump(scaler, SCALER_PATH)

        del model
        del scaler
        keras.backend.clear_session()
        gc.collect()

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        os.rename(temp_model_path, MODEL_PATH)

        model, scaler = load_assets()
        print("Model and scaler reloaded successfully after retraining.")
    except Exception as e:
        raise RuntimeError(f"Failed to save/reload model after retraining: {e}")