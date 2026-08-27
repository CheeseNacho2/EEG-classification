import tensorflow as tf
from tensorflow import keras
import numpy as np
import mne
import joblib
import os
from scipy.signal import welch

mne.set_log_level('WARNING')

# Filtering out stress-relevant channel names
STRESS_CHANNELS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4',
                   'EEG Fz', 'EEG T3', 'EEG T4', 'EEG P3', 'EEG P4']

#Bands affected the most by stress related brain activity and their signal frequencies
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30)
    }

EPOCH_DURATION = 2.0
OVERLAP        = 1.0

#Loading in data from .edf files - mne.io processes them and returns relevant data
def load_and_filter(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    
    available = [ch for ch in STRESS_CHANNELS if ch in raw.ch_names]
    raw.pick(available)
    
    # Read the unit multiplier from the file metadata
    unit_mul = raw.info['chs'][0]['unit_mul']
    
    # Convert to microvolts based on whatever unit the file uses
    # MNE unit_mul values: 0=none, -6=micro, -3=milli
    if unit_mul == 0:    # Stored in Volts → multiply by 1e6
        scale = 1e6
    elif unit_mul == -3: # Stored in millivolts → multiply by 1e3
        scale = 1e3
    elif unit_mul == -6: # Already in microvolts → no change
        scale = 1.0
    else:
        scale = 1.0
        print(f"Warning: Unknown unit multiplier {unit_mul} in {filepath}")
    
    raw.apply_function(lambda x: x * scale, picks='all')
    
    raw.filter(0.5, 45, verbose=False)
    return raw


# Epoching
def epoch_signal(raw):
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_DURATION,
        overlap=OVERLAP,
        verbose=False
    )
    return epochs.get_data()  # Shape: (n_epochs, 9 channels, 1000 samples)

#Based on extracted frequencies and band names
def extract_band_power(epoch, sfreq=500):
    features = []

    for channel in epoch:  # Loop over each of the 9 channels
        # Welch method estimates power at each frequency and returns array of frequncies and avg. power
        freqs, psd = welch(channel, sfreq, nperseg=sfreq*2)

        for band_name, (low, high) in BANDS.items():
            # Find which frequencies fall within this band
            band_mask = (freqs >= low) & (freqs <= high)
            # Average power within that band
            band_power = np.mean(psd[band_mask])
            features.append(band_power)

    # Frontal asymmetry — F4 minus F3 alpha power (key stress marker) to add to the final array
    f3_idx = STRESS_CHANNELS.index('EEG F3')
    f4_idx = STRESS_CHANNELS.index('EEG F4')

    _, psd_f3 = welch(epoch[f3_idx], sfreq, nperseg=sfreq*2)
    _, psd_f4 = welch(epoch[f4_idx], sfreq, nperseg=sfreq*2)

    freqs, _ = welch(epoch[f3_idx], sfreq, nperseg=sfreq*2)
    alpha_mask = (freqs >= 8) & (freqs <= 13)

    frontal_asymmetry = np.mean(psd_f4[alpha_mask]) - np.mean(psd_f3[alpha_mask])
    features.append(frontal_asymmetry)

    return np.array(features)

def get_feature_names():
    names = []
    for ch in STRESS_CHANNELS:
        for band in BANDS.keys():
            names.append(f"{ch}_{band}")
    names.append("frontal_asymmetry")
    return names
