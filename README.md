# EEG-classifier

## General Description
This classifier uses TensorFlow's neural networks to determine whether a subject is in a calm or stressed state. The input is via EDF files which contain brain signals obtained from EEG diagnostics. Subjects undergoing EEG have electrodes placed on specific parts of their head to measure the local electrical response to various stimuli. Some parts of the brain are more sensitive to stress and the neural network of this model is trained by focusing on changes in those regions.

The brain regions most relevant to stress detection are the frontal and prefrontal cortex, which are responsible for emotional regulation and decision making. Research has shown that stress causes measurable changes in specific frequency bands of the EEG signal. Alpha waves (8-13 Hz) usually decrease under stress, while beta waves (13-30 Hz) increase. Theta waves (4-8 Hz) in the frontal regions also increase during heavy mental workload. A particularly reliable stress marker is frontal alpha asymmetry — the difference in alpha power between the left (F3) and right (F4) frontal electrodes — where greater right-sided activity is associated with stress and negative affect.

## Methodology
Using the MNE Python library, the data from EDF files can be extracted and transformed into arrays. After filtering the arrays for the data of the regions that matter (based on the electrode and frequency), the arrays can be fed to train the neural network. The neural network needs labeled data to be trained. After training, the model is ready to classify an EEG containing file without the label.

## How to Use
1. First use `EEG classification.py` to create a trained model with a scaler (used to normalize the scale of input features). For that we need datasets of EDF files and this specific code needs them labeled as `Subject[number]_[1 or 2]` — 1 for the calm state and 2 for the stressed state. The relative path to the datasets is by default named as `eegsignals/eeg-during-mental-arithmetic-tasks-1.0.0` but can be changed in the code. An example folder is in this repository.
2. After the `model.keras` and `scaler.pkl` files are created, the app can run. You can simply open the app using `run.py` or start up `api.py` and then open localhost to access the web version.
3. Both versions can predict independent EDF files given that they have data in all the necessary channels.

## Credits
- Dataset: PhysioNet EEGMAT — Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, Shpenkov O. Electroencephalograms during Mental Arithmetic Task Performance. Data. 2019
- Frontal alpha asymmetry as a stress marker: Davidson, R.J. (1998). Anterior electrophysiological asymmetries, emotion, and depression. Psychophysiology, 35(5), 607-614
- EEG frequency bands and stress: Oken, B.S., Salinsky, M.C., & Elsas, S.M. (2006). Vigilance, alertness, or sustained attention: physiological basis and measurement. Clinical Neurophysiology, 117(9), 1885-1901
- Alpha wave suppression under stress: Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance. Brain Research Reviews, 29(2-3), 169-195
- Built with MNE-Python, TensorFlow, FastAPI, PyQt6, scikit-learn, Chart.js
- Developed with assistance from Claude (Anthropic)
