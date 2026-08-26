import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import mne
import os
import glob
from scipy.signal import welch
from scipy.signal import butter, filtfilt    
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

tf.random.set_seed(42)
np.random.seed(42)

data_folder = os.path.join(os.path.dirname(__file__),
                           "eegsignals",
                           "eeg-during-mental-arithmetic-tasks-1.0.0")

edf_files = glob.glob(os.path.join(data_folder, "Subject*_*.edf"))
edf_files.sort()

print(f"Found {len(edf_files)} EDF files across {len(edf_files)//2} subjects")

# ── Auto Labeling ────────────────────────────────────────────────
def get_label(filepath):
    filename = os.path.basename(filepath)
    if filename.endswith("_2.edf"):
        return 1  # Stress
    elif filename.endswith("_1.edf"):
        return 0  # Calm/Baseline

dataset = [(f, get_label(f)) for f in edf_files]

# Verify
for filepath, label in dataset:
    print(f"{os.path.basename(filepath)} -> {'Stress' if label == 1 else 'Calm'}")

# ── Stress-Relevant Channels ─────────────────────────────────────
STRESS_CHANNELS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4',
                   'EEG Fz', 'EEG T3', 'EEG T4', 'EEG P3', 'EEG P4']

# ── Loading & Filtering ──────────────────────────────────────────
def load_and_filter(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    
    available = [ch for ch in STRESS_CHANNELS if ch in raw.ch_names]
    raw.pick(available)  # Updated from pick_channels
    
    raw.filter(0.5, 45, verbose=False)
    
    return raw

# Testing
#raw = load_and_filter(dataset[0][0])
#print(f"Channels: {raw.ch_names}")
#print(f"Sample rate: {raw.info['sfreq']} Hz")
#print(f"Duration: {raw.times[-1]:.1f} seconds")

# ── Epoching ─────────────────────────────────────────────────────
EPOCH_DURATION = 2.0   # seconds per epoch
OVERLAP = 0.5          # 50% overlap between epochs

def epoch_signal(raw):
    sfreq = raw.info['sfreq']                    # 500 Hz
    epoch_samples = int(EPOCH_DURATION * sfreq)  # 1000 samples per epoch
    step_samples = int(epoch_samples * (1 - OVERLAP))  # 500 samples step

    data = raw.get_data()  # Shape: (9 channels, total_samples)
    epochs = []

    for start in range(0, data.shape[1] - epoch_samples, step_samples):
        epoch = data[:, start:start + epoch_samples]
        epochs.append(epoch)

    return np.array(epochs)  # Shape: (n_epochs, 9 channels, 1000 samples)

# Testing
#epochs = epoch_signal(raw)
#print(f"Epochs shape: {epochs.shape}")
#print(f"Number of epochs from first file: {epochs.shape[0]}")

BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30)
}

def extract_band_power(epoch, sfreq=500):
    """
    For each channel, calculate the power in each frequency band.
    Returns a 1D feature vector.
    """
    features = []

    for channel in epoch:  # Loop over each of the 9 channels
        # Welch method estimates power at each frequency
        freqs, psd = welch(channel, sfreq, nperseg=sfreq*2)

        for band_name, (low, high) in BANDS.items():
            # Find which frequencies fall within this band
            band_mask = (freqs >= low) & (freqs <= high)
            # Average power within that band
            band_power = np.mean(psd[band_mask])
            features.append(band_power)

    # Frontal asymmetry — F4 minus F3 alpha power (key stress marker)
    f3_idx = STRESS_CHANNELS.index('EEG F3')
    f4_idx = STRESS_CHANNELS.index('EEG F4')

    _, psd_f3 = welch(epoch[f3_idx], sfreq, nperseg=sfreq*2)
    _, psd_f4 = welch(epoch[f4_idx], sfreq, nperseg=sfreq*2)

    freqs, _ = welch(epoch[f3_idx], sfreq, nperseg=sfreq*2)
    alpha_mask = (freqs >= 8) & (freqs <= 13)

    frontal_asymmetry = np.mean(psd_f4[alpha_mask]) - np.mean(psd_f3[alpha_mask])
    features.append(frontal_asymmetry)

    return np.array(features)

# Testing
#sample_features = extract_band_power(epochs[0])
#print(f"Features per epoch: {sample_features.shape[0]}")
#print(f"Breakdown: 9 channels × 3 bands = 27, + 1 frontal asymmetry = 28")

def build_dataset(dataset):
    all_features = []
    all_labels = []

    for filepath, label in dataset:
        print(f"Processing {os.path.basename(filepath)}...")
        raw = load_and_filter(filepath)
        epochs = epoch_signal(raw)
        for epoch in epochs:
            features = extract_band_power(epoch)
            all_features.append(features)
            all_labels.append(label)

    X = np.array(all_features)
    y = np.array(all_labels)

    #print(f"\nDataset built!")
    #print(f"X shape: {X.shape}")
    #print(f"y shape: {y.shape}")
    #print(f"Calm epochs:   {np.sum(y == 0)}")
    #print(f"Stress epochs: {np.sum(y == 1)}")

    return X, y

X, y = build_dataset(dataset)

# Separate classes
X_calm   = X[y == 0]
X_stress = X[y == 1]

# Undersample calm to match stress count
X_calm_balanced = resample(X_calm, 
                           n_samples=len(X_stress),
                           random_state=42)

# Recombine
X_balanced = np.vstack([X_calm_balanced, X_stress])
y_balanced = np.hstack([np.zeros(len(X_stress)), 
                        np.ones(len(X_stress))])

#print(f"Balanced dataset:")
#print(f"Calm epochs:   {np.sum(y_balanced == 0)}")
#print(f"Stress epochs: {np.sum(y_balanced == 1)}")
#print(f"Total:         {len(y_balanced)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)

joblib.dump(scaler, 'scaler.pkl')

# ── Build Model ───────────────────────────────────────────────────
def build_model(input_shape):
    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # First dense block
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second dense block
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third dense block
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output — single neuron, stress probability 0-1
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = build_model(input_shape=(28,))
model.summary()
os.makedirs('models', exist_ok=True)

# ── Callbacks ─────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='models/model.keras',
        monitor='val_loss',
        save_best_only=True
    )
]

# ── Train ─────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=110,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed classification report
from sklearn.metrics import classification_report, confusion_matrix

y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Calm', 'Stress']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Calm', 'Stress'],
            yticklabels=['Calm', 'Stress'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png')
plt.show()

# ── Save Model ────────────────────────────────────────────────────
model.save('models/model.keras')
print("Model saved!")