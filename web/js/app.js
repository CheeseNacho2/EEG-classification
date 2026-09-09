// ── API URL ───────────────────────────────────────────────────────
const API_URL = 'http://localhost:8000';

// ── State ─────────────────────────────────────────────────────────
let selectedFile    = null;
let epochChart      = null;
let importanceChart = null;

// ── Elements ──────────────────────────────────────────────────────
const fileInput     = document.getElementById('file-input');
const uploadArea    = document.getElementById('upload-area');
const fileNameLabel = document.getElementById('file-name');
const predictBtn    = document.getElementById('predict-btn');
const loadingCard   = document.getElementById('loading-card');
const loadingText   = document.getElementById('loading-text');
const resultCard    = document.getElementById('result-card');
const resultBadge   = document.getElementById('result-badge');
const resultLabel   = document.getElementById('result-label');
const resultProb    = document.getElementById('result-prob');
const graphsRow     = document.getElementById('graphs-row');
const retrainCard   = document.getElementById('retrain-card');
const yesBtnEl      = document.getElementById('yes-btn');
const noBtnEl       = document.getElementById('no-btn');
const retrainStatus = document.getElementById('retrain-status');
const banner        = document.getElementById('banner');

// ── API Health Check on Load ──────────────────────────────────────
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`, {
            signal: AbortSignal.timeout(3000)
        });

        if (response.ok) {
            const data = await response.json();
            if (!data.model_loaded) {
                showBanner(
                    "Warning: Model is not loaded. Please run train.py first.",
                    "warning"
                );
            }
        } else {
            showBanner(
                "API returned an unexpected status. Some features may not work.",
                "warning"
            );
        }
    } catch (err) {
        if (err.name === 'TimeoutError') {
            showBanner(
                "Cannot connect to the API — request timed out. " +
                "Please make sure api.py is running.",
                "error"
            );
        } else {
            showBanner(
                "Cannot connect to the API. " +
                "Please make sure api.py is running.",
                "error"
            );
        }
        predictBtn.disabled = true;
    }
});

// ── File Handling ─────────────────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#4f46e5';
    uploadArea.style.background  = '#f5f3ff';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#c7d2fe';
    uploadArea.style.background  = '';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#c7d2fe';
    uploadArea.style.background  = '';
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
});

function handleFileSelect(file) {
    // Validate extension
    if (!file.name.toLowerCase().endsWith('.edf')) {
        showBanner('Please upload a valid .edf file.', 'error');
        return;
    }

    // Validate file is not empty
    if (file.size === 0) {
        showBanner('The selected file is empty.', 'error');
        return;
    }

    selectedFile              = file;
    fileNameLabel.textContent = `Selected: ${file.name}`;
    predictBtn.disabled       = false;
    hideBanner();

    // Reset previous results
    hide(resultCard);
    hide(graphsRow);
    hide(retrainCard);
    retrainStatus.textContent = '';
}

// ── Prediction ────────────────────────────────────────────────────
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    predictBtn.disabled     = true;
    loadingText.textContent = 'Analysing EEG signal...';
    show(loadingCard);
    hide(resultCard);
    hide(graphsRow);
    hide(retrainCard);
    hideBanner();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body:   formData,
            signal: AbortSignal.timeout(60000)
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${response.status})`);
        }

        const result = await response.json();

        if (!result.label || result.probability === undefined) {
            throw new Error('Invalid response from server.');
        }

        showResult(result);
        await loadFeatureImportance();

    } catch (err) {
        if (err.name === 'TimeoutError') {
            showBanner(
                'Prediction timed out. The file may be too large or the server is busy.',
                'error'
            );
        } else if (err.message.includes('Failed to fetch')) {
            showBanner(
                'Cannot connect to the API. Please make sure api.py is running.',
                'error'
            );
        } else {
            showBanner(`Prediction error: ${err.message}`, 'error');
        }
    } finally {
        hide(loadingCard);
        predictBtn.disabled = false;
    }
});

// ── Show Result ───────────────────────────────────────────────────
function showResult(result) {
    const label = result.label;
    const prob  = (result.probability * 100).toFixed(1);

    resultLabel.textContent = label;
    resultProb.textContent  = `${prob}%`;
    resultBadge.className   = `result-badge ${label.toLowerCase()}`;

    show(resultCard);

    try {
        plotEpochChart(result.per_epoch);
        show(graphsRow);
    } catch (err) {
        console.error('Failed to plot epoch chart:', err);
        showBanner('Result received but epoch chart failed to render.', 'warning');
    }

    // Reset feedback section
    yesBtnEl.disabled         = false;
    noBtnEl.disabled          = false;
    yesBtnEl.classList.remove('hidden');
    noBtnEl.classList.remove('hidden');
    retrainStatus.textContent = '';
    retrainStatus.style.color = '';
    show(retrainCard);
}

// ── Epoch Chart ───────────────────────────────────────────────────
function plotEpochChart(epochProbs) {
    if (!epochProbs || epochProbs.length === 0) {
        throw new Error('No epoch data to plot.');
    }

    const n      = epochProbs.length;
    const factor = 10;
    const xFine  = [];
    const yFine  = [];

    for (let i = 0; i < (n - 1) * factor; i++) {
        const t    = i / factor;
        const idx  = Math.floor(t);
        const frac = t - idx;
        xFine.push(t);
        yFine.push(epochProbs[idx] * (1 - frac) + epochProbs[idx + 1] * frac);
    }

    const stressData = yFine.map(v => v >= 0.5 ? v : null);
    const calmData   = yFine.map(v => v  < 0.5 ? v : null);

    if (epochChart) epochChart.destroy();

    const ctx = document.getElementById('epoch-chart').getContext('2d');
    epochChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: xFine.map(x => x.toFixed(1)),
            datasets: [
                {
                    label:           'Stress',
                    data:            stressData,
                    borderColor:     '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    fill:            true,
                    pointRadius:     0,
                    tension:         0.3,
                    spanGaps:        false
                },
                {
                    label:           'Calm',
                    data:            calmData,
                    borderColor:     '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.15)',
                    fill:            true,
                    pointRadius:     0,
                    tension:         0.3,
                    spanGaps:        false
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        title: (items) => `Epoch ${Math.floor(items[0].parsed.x)}`,
                        label: (item) => {
                            if (item.parsed.y === null) return null;
                            const state = item.parsed.y >= 0.5 ? 'Stress' : 'Calm';
                            return `${state}: ${(item.parsed.y * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { maxTicksLimit: 10 },
                    title: { display: true, text: 'Epoch' }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Probability' },
                    ticks: { callback: v => `${(v * 100).toFixed(0)}%` }
                }
            }
        }
    });
}

// ── Feature Importance ────────────────────────────────────────────
async function loadFeatureImportance() {
    try {
        const response = await fetch(`${API_URL}/feature-importance`, {
            signal: AbortSignal.timeout(10000)
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${response.status})`);
        }

        const data  = await response.json();

        if (!data.features || data.features.length === 0) {
            throw new Error('No feature importance data returned.');
        }

        const top10  = data.features.slice(0, 10);
        const bandSymbols = { theta: 'θ', alpha: 'α', beta: 'β' };

        const names  = top10.map(f => {
            let name = f.name.replace('EEG ', '');
            Object.entries(bandSymbols).forEach(([band, symbol]) => {
                name = name.replace(`_${band}`, ` ${symbol}`);
            });
            return name;
        });

        const scores = top10.map(f => f.importance);
        const colors = scores.map((_, i) =>
            i < 3 ? 'rgba(79, 70, 229, 0.85)' : 'rgba(79, 70, 229, 0.35)'
        );

        if (importanceChart) importanceChart.destroy();

        const ctx = document.getElementById('importance-chart').getContext('2d');
        importanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: names,
                datasets: [{
                    label:           'Importance',
                    data:            scores,
                    backgroundColor: colors,
                    borderRadius:    4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: item => `${(item.parsed.x * 100).toFixed(2)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Importance' },
                        ticks: { callback: v => `${(v * 100).toFixed(1)}%` }
                    },
                    y: { ticks: { font: { size: 11 } } }
                }
            }
        });

    } catch (err) {
        if (err.name === 'TimeoutError') {
            showBanner('Feature importance timed out.', 'warning');
        } else {
            showBanner(`Feature importance unavailable: ${err.message}`, 'warning');
        }
    }
}

// ── Retraining ────────────────────────────────────────────────────
yesBtnEl.addEventListener('click', () => {
    yesBtnEl.classList.add('hidden');
    noBtnEl.classList.add('hidden');
    retrainStatus.textContent = '✓ Thank you for your feedback!';
    retrainStatus.style.color = '#16a34a';
});

noBtnEl.addEventListener('click', async () => {
    const currentLabel = resultLabel.textContent;
    const correctLabel = currentLabel === 'Stress' ? 0 : 1;
    await retrain(correctLabel);
});

async function retrain(label) {
    if (!selectedFile) return;

    yesBtnEl.disabled       = true;
    noBtnEl.disabled        = true;
    loadingText.textContent = 'Retraining model on new data...';
    show(loadingCard);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('label', label);

    try {
        const response = await fetch(`${API_URL}/retrain`, {
            method: 'POST',
            body:   formData,
            signal: AbortSignal.timeout(120000)
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${response.status})`);
        }

        yesBtnEl.classList.add('hidden');
        noBtnEl.classList.add('hidden');
        retrainStatus.textContent = '✓ Thank you for your feedback!';
        retrainStatus.style.color = '#16a34a';

    } catch (err) {
        if (err.name === 'TimeoutError') {
            retrainStatus.textContent = 'Retraining timed out. Please try again.';
        } else if (err.message.includes('Failed to fetch')) {
            retrainStatus.textContent = 'Cannot connect to API. Please make sure api.py is running.';
        } else {
            retrainStatus.textContent = `Retraining error: ${err.message}`;
        }
        retrainStatus.style.color = '#dc2626';
        yesBtnEl.disabled         = false;
        noBtnEl.disabled          = false;
    } finally {
        hide(loadingCard);
    }
}

// ── Banner ────────────────────────────────────────────────────────
function showBanner(message, type = 'error') {
    const styles = {
        error:   { bg: '#fee2e2', color: '#dc2626', border: '#fecaca' },
        warning: { bg: '#fef9c3', color: '#92400e', border: '#fde68a' },
        success: { bg: '#dcfce7', color: '#16a34a', border: '#bbf7d0' }
    };
    const s = styles[type] || styles.error;
    banner.textContent   = message;
    banner.style.display = 'block';
    banner.style.background   = s.bg;
    banner.style.color        = s.color;
    banner.style.border       = `1px solid ${s.border}`;
}

function hideBanner() {
    banner.style.display = 'none';
}

// ── Helpers ───────────────────────────────────────────────────────
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden');    }