import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import glob   
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils import (STRESS_CHANNELS, BANDS, EPOCH_DURATION, OVERLAP, load_and_filter, epoch_signal, extract_band_power)

tf.random.set_seed(42)
np.random.seed(42)

#Path to the folder with .edf files
data_folder = os.path.join(os.path.dirname(__file__), "eegsignals", "eeg-during-mental-arithmetic-tasks-1.0.0")

#List of the .edf file names sorted
edf_files = glob.glob(os.path.join(data_folder, "Subject*_*.edf"))
edf_files.sort()

# Assign labels from filename automatically
def get_label(filepath):
    filename = os.path.basename(filepath)
    if filename.endswith("_2.edf"):
        return 1  # Stress
    elif filename.endswith("_1.edf"):
        return 0  # Calm/Baseline

dataset = [(f, get_label(f)) for f in edf_files]

# Verify
#for filepath, label in dataset:
#    print(f"{os.path.basename(filepath)} -> {'Stress' if label == 1 else 'Calm'}")

# Testing
#epochs = epoch_signal(raw)
#print(f"Epochs shape: {epochs.shape}")
#print(f"Number of epochs from first file: {epochs.shape[0]}")


# Testing
#sample_features = extract_band_power(epochs[0])
#print(f"Features per epoch: {sample_features.shape[0]}")
#print(f"Breakdown: 9 channels × 3 bands = 27, + 1 frontal asymmetry = 28")


#Building the dataset using processed data
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

# Undersample calm to match stress count - to prevent prediction bias
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


#Splitting samples into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)

#Normalisation due to diferences in signal strengths in different regions
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

#  Build the CNN
def build_model(input_shape):
    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # First Hidden Layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second Hidden Layer
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third Hidden Layer
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output
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

# Callbacks - In case accuracy never reaches the highest point again
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

# Training
history = model.fit(
    X_train, y_train,
    epochs=110,
    batch_size=32,
    validation_data=(X_test, y_test),
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