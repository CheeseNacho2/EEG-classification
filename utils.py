import numpy as np
import mne
from scipy.signal import welch
import warnings

mne.set_log_level('WARNING')

# Filtering out stress-relevant channel names
STRESS_CHANNELS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4',
                   'EEG Fz', 'EEG T3', 'EEG T4', 'EEG P3', 'EEG P4']

# Due to channel naming in Emotiv devices / 10-10 systems
CHANNEL_ALIASES = {
    'af3': 'EEG F3',
    'af4': 'EEG F4',
    't7':  'EEG T3',
    't8':  'EEG T4',
    'p7':  'EEG P3',
    'p8':  'EEG P4',
    'pz':  'EEG Fz',
    'f3':  'EEG F3',
    'f4':  'EEG F4',
    'fp1': 'EEG Fp1',
    'fp2': 'EEG Fp2',
    'fz':  'EEG Fz',
    't3':  'EEG T3',
    't4':  'EEG T4',
    'p3':  'EEG P3',
    'p4':  'EEG P4',
}

BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30)
}

EPOCH_DURATION = 2.0
OVERLAP        = 1.0


def load_and_filter(filepath):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Invalid measurement date")
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read EDF file '{filepath}': {e}")

    rename_map       = {}
    assigned_targets = set()

    for ch in raw.ch_names:
        clean_name = ch.replace('EEG', '').strip().lower()
        if clean_name in CHANNEL_ALIASES:
            target_name = CHANNEL_ALIASES[clean_name]
            if target_name not in assigned_targets:
                rename_map[ch] = target_name
                assigned_targets.add(target_name)

    if rename_map:
        raw.rename_channels(rename_map)

    available = [ch for ch in STRESS_CHANNELS if ch in raw.ch_names]

    if not available:
        raise ValueError(
            f"No stress-relevant channels found in '{filepath}'. "
            f"File contains: {raw.ch_names[:10]}... "
            f"Expected one or more of: {STRESS_CHANNELS}"
        )

    raw.pick(available)

    unit_mul = raw.info['chs'][0].get('unit_mul', 0)
    if unit_mul == 0:
        scale = 1e6
    elif unit_mul == -3:
        scale = 1e3
    elif unit_mul == -6:
        scale = 1.0
    else:
        scale = 1.0
        print(f"Warning: Unknown unit multiplier {unit_mul} in '{filepath}'")

    raw.apply_function(lambda x: x * scale, picks='all')
    raw.filter(0.5, 45, verbose=False)
    return raw



# Epoching
def epoch_signal(raw):
    try:
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=EPOCH_DURATION,
            overlap=OVERLAP,
            verbose=False
        )
        data = epochs.get_data()
    except Exception as e:
        raise RuntimeError(f"Failed to epoch signal: {e}")

    if data.shape[0] == 0:
        raise ValueError(
            f"No epochs were created. Recording may be too short. "
            f"Minimum duration required: {EPOCH_DURATION} seconds."
        )

    return data  # Shape: (n_epochs, 9 channels, 1000 samples)

#Based on extracted frequencies and band names
def extract_band_power(epoch, sfreq=500):
    if epoch.ndim != 2:
        raise ValueError(
            f"Expected 2D epoch array (channels, samples), got shape {epoch.shape}"
        )

    if epoch.shape[0] == 0:
        raise ValueError("Epoch has no channels.")

    if epoch.shape[1] == 0:
        raise ValueError("Epoch has no samples.")

    features = []

    for i, channel in enumerate(epoch):
        try:
            freqs, psd = welch(channel, sfreq, nperseg=sfreq * 2)
        except Exception as e:
            raise RuntimeError(f"Welch PSD failed on channel {i}: {e}")

        for band_name, (low, high) in BANDS.items():
            band_mask  = (freqs >= low) & (freqs <= high)
            if not np.any(band_mask):
                raise ValueError(
                    f"No frequency bins found for band '{band_name}' "
                    f"({low}-{high} Hz) at sfreq={sfreq} Hz."
                )
            band_power = np.mean(psd[band_mask])
            features.append(band_power)

    # Frontal asymmetry
    try:
        f3_idx = STRESS_CHANNELS.index('EEG F3')
        f4_idx = STRESS_CHANNELS.index('EEG F4')

        if f3_idx >= epoch.shape[0] or f4_idx >= epoch.shape[0]:
            # F3 or F4 not available — append zero as placeholder
            features.append(0.0)
        else:
            _, psd_f3  = welch(epoch[f3_idx], sfreq, nperseg=sfreq * 2)
            _, psd_f4  = welch(epoch[f4_idx], sfreq, nperseg=sfreq * 2)
            freqs, _   = welch(epoch[f3_idx], sfreq, nperseg=sfreq * 2)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            frontal_asymmetry = np.mean(psd_f4[alpha_mask]) - np.mean(psd_f3[alpha_mask])
            features.append(frontal_asymmetry)
    except Exception as e:
        raise RuntimeError(f"Failed to compute frontal asymmetry: {e}")

    features = np.array(features)

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError(
            "Feature extraction produced NaN or Inf values. "
            "Check signal quality or filtering settings."
        )

    return features

def get_feature_names():
    names = []
    for ch in STRESS_CHANNELS:
        for band in BANDS.keys():
            names.append(f"{ch}_{band}")
    names.append("frontal_asymmetry")
    return names
