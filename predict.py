import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
import os
from utils import (STRESS_CHANNELS, BANDS, EPOCH_DURATION,
                   OVERLAP, load_and_filter, epoch_signal,
                   extract_band_power, get_feature_names)

# ── Paths ─────────────────────────────────────────────────────────
MODEL_PATH  = 'models/model.keras'
SCALER_PATH = 'models/scaler.pkl'

# ── Load Model & Scaler ───────────────────────────────────────────
def load_assets():
    model  = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ── Predict ───────────────────────────────────────────────────────
def predict_stress(filepath):
    model, scaler = load_assets()

    # Preprocess
    raw    = load_and_filter(filepath)
    epochs = epoch_signal(raw)

    # Extract features from each epoch
    features = np.array([extract_band_power(epoch) for epoch in epochs])

    # Scale using saved scaler
    features_scaled = scaler.transform(features)

    # Predict each epoch
    epoch_probs = model.predict(features_scaled, verbose=0).flatten()

    # Average across all epochs for final result
    mean_prob = float(np.mean(epoch_probs))
    label     = 'Stress' if mean_prob >= 0.5 else 'Calm'

    return {
        'label':       label,
        'probability': round(mean_prob, 4),
        'per_epoch':   epoch_probs.tolist(),
        'n_epochs':    len(epoch_probs)
    }

# ── Retrain ───────────────────────────────────────────────────────
def retrain(new_filepath, true_label):
    """
    true_label: 0 = Calm, 1 = Stress
    Called from UI when user submits new labelled EEG data.
    """
    model, scaler = load_assets()

    # Preprocess new file
    raw    = load_and_filter(new_filepath)
    epochs = epoch_signal(raw)

    # Extract and scale features
    features        = np.array([extract_band_power(epoch) for epoch in epochs])
    features_scaled = scaler.transform(features)
    labels          = np.full(len(features), true_label)

    # Fine-tune model on new data
    model.fit(
        features_scaled, labels,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Save updated model
    model.save(MODEL_PATH)
    print(f"Model retrained on {os.path.basename(new_filepath)} and saved.")