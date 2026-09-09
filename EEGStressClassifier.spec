import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Dynamically resolve root directory
ROOT_DIR = os.path.abspath('.')

mne_datas, mne_binaries, mne_hiddenimports       = collect_all('mne')
tf_datas, tf_binaries, tf_hiddenimports          = collect_all('tensorflow')
keras_datas, keras_binaries, keras_hiddenimports = collect_all('keras')

a = Analysis(
    ['run.py'],
    pathex=[ROOT_DIR],
    binaries=[
        *mne_binaries,
        *tf_binaries,
        *keras_binaries,
    ],
    datas=[
        ('models/model.keras', 'models'),
        ('models/scaler.pkl',  'models'),
        ('desktop/assets/brain.png',  'desktop/assets'),
        ('desktop/assets/splash.gif', 'desktop/assets'),

        ('api.py',         '.'),
        ('predict.py',     '.'),
        ('utils.py',       '.'),
        ('desktop/app.py', 'desktop'),

        *mne_datas,
        *tf_datas,
        *keras_datas,
    ],
    hiddenimports=[
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'sklearn',
        'sklearn.utils',
        'sklearn.preprocessing',
        'joblib',
        'scipy',
        'scipy.signal',
        'mne',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_qt5agg',
        *mne_hiddenimports,
        *tf_hiddenimports,
        *keras_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'eegsignals',
        'train',
        'web',
        'matplotlib.tests',
        'numpy.tests',
        'tensorflow_io_gcs_filesystem',
        'tensorboard',
        'torch',
        'jax',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EEGStressClassifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon='desktop/assets/icon.ico'  
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='EEGStressClassifier'
)