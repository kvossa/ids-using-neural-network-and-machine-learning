# Intrusion Detection System — Hybrid Neural Network (CNN + LSTM + Autoencoder)

An Intrusion Detection System (IDS) for network traffic classification built on a hybrid CNN + LSTM + Autoencoder architecture. It supports a **two-stage pipeline for CIC-IDS2017** (binary Normal vs Attack gate, then attack-group classification) and a **single-stage pipeline for UNSW-NB15** (confusion-based multiclass groups), plus **live/streaming detection** and a lab environment for real traffic.

## Features

- **Hybrid architecture** — CNN (spatial) + LSTM (temporal) branches fused with an autoencoder bottleneck; outputs both classification and reconstruction (anomaly signal).
- **Two-stage CIC pipeline** — stage 1 binary gate with calibrated threshold (`threshold.json`); stage 2 maps the 7 attack classes to Flood/Rare groups.
- **Single-stage UNSW pipeline** — confusion-based grouping of the 10 UNSW classes (7 groups), including Normal.
- **Class imbalance handling** — binary focal loss + balanced datasets; multiclass SMOTE/oversampling + categorical focal loss with adaptive alpha; class weights for fine-tuning.
- **Threshold calibration** — calibrated stage-1 attack threshold saved alongside the model.
- **Batch and live inference** — flow CSVs (`infer_batch.py`) or streaming from an interface (`infer_live.py`).
- **Fine-tuning** on labeled real traffic per dataset/head.
- **Lab environment** — traffic orchestration, windowed capture, ground-truth evaluation (`experiment/`).

## Architecture

The model (`src/model/model.py`) takes three inputs and produces two outputs:

```
ae_input  ──► Dense(48) ─► Dense(24) ─► Dense(48) ─► sigmoid ──► reconstruction
                  └──► StopGradient (bottleneck features)
cnn_input ──► Conv1D(64) ─► MaxPool ─► Conv1D(128) ─► MaxPool ─► GlobalAvgPool ─► Dropout
lstm_input─► LayerNorm ─► LSTM(128) ─► Dropout
                │                              │
                └──────────── Concatenate (feature fusion) ─────────────┘
                                            │
                              classification head (standard | attention)
                                            │
                                       classification
```

- `ae_input` — last flow of the window (feature vector)
- `cnn_input` / `lstm_input` — the full windowed sequence
- `classification` — class/group probabilities
- `reconstruction` — autoencoder reconstruction (MSE used as an anomaly signal)

The classification head supports `standard` and `attention` depths. Preprocessing is a fixed pipeline chain (`DataCleaner → FeatureExtraction → ColumnDropper → CategoricalEncoder → Imputer → LogTransformer → RobustScaler → MinMaxScaler`) and **windowing is required** before training (`src/preprocessing/windowing/windowing.py`).

**CIC two-stage flow:** stage 1 gates each window as `BENIGN` or `Attack` (calibrated threshold) → flagged windows go to stage 2 → `Flood` / `Rare` group.

## Datasets

| Dataset | Format | Label column | Normal class | Features | Window size |
|---------|--------|--------------|--------------|----------|-------------|
| CIC-IDS2017 | Parquet | `attack_type` | `BENIGN` | 41 | 5 |
| UNSW-NB15 | CSV | `attack_cat` | `Normal` | 193 | 10 |

Raw data goes in `data/raw/`; processed splits in `data/processed/{CIC-IDS2017,UNSW-NB15}/splits/`.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/config.py` | Centralized artifact/data paths |
| `src/model/` | Hybrid CNN + LSTM + Autoencoder architecture |
| `src/preprocessing/` | Data pipeline + windowing |
| `src/training/` | Training scripts (CIC two-stage, UNSW single-stage, fine-tuning) |
| `src/inference/` | Batch + streaming inference classes (`cic/`, `unsw/` subpackages) |
| `src/grouping/` | Class → group definitions |
| `src/utils/` | Helpers (visualization, logging, splitting, scoring, gap analysis) |
| `src/cicflowmeter/` | Vendored CICFlowMeter (self-contained) |
| `experiment/` | Lab environment (orchestrator, capture, batch/live inference, evaluation) |
| `notebooks/` | EDA and model testing notebooks |
| `reports/` | Metrics and figures |

## Installation

Python 3.9 recommended.

```bash
pip install -r requirements.txt
```

Extra for the lab: `tcpdump` (root or `CAP_NET_RAW`), `tshark` for UNSW live capture, and optionally `java` for CICFlowMeter. See [`experiment/README.md`](experiment/README.md).

## Usage

### Preprocessing

Build and save the preprocessing pipelines:

```bash
python -m src.preprocessing.main
```

Exports to `models/preprocessing/{binary,multiclass}/{cic,unsw}/`.

### Training

```bash
python src/training/cic/stage_1.py      # CIC stage 1 — binary Normal vs Attack + threshold calibration
python src/training/cic/stage_2.py      # CIC stage 2 — Flood vs Rare grouping
python src/training/unsw/single_stage.py  # UNSW single-stage multiclass (SMOTE + focal loss)
python -m src.training.fine_tune --dataset {CIC,UNSW}  # fine-tune heads on labeled real traffic
```

### Inference

Batch inference on a flows CSV:

```bash
python experiment/scripts/infer_batch.py --config experiment/config/lab.json --flows-csv <csv>
```

Live/streaming detection:

```bash
python experiment/scripts/infer_live.py --mode {cic,unsw} [--interface eth0]
```

### Lab experiments

Traffic orchestration, windowed capture, and evaluation against temporal ground truth are documented in [`experiment/README.md`](experiment/README.md).

## Results

Key test-set metrics (full reports in `reports/metrics/`):

| Model | Task | Accuracy | Notes |
|-------|------|----------|-------|
| CIC stage 1 | Binary gate | 0.74 | Attack F1 0.84 |
| CIC stage 2 (bruterare) | Flood vs Rare | 0.78 | Flood F1 0.69, Rare F1 0.83 |
| UNSW (confusion groups, baseline) | 7 multiclass groups | 0.78 | Normal F1 0.89, macro F1 0.72 |

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/eda_cic.ipynb` | CIC-IDS2017 exploratory analysis |
| `notebooks/eda_unsw.ipynb` | UNSW-NB15 exploratory analysis |
| `notebooks/features.ipynb` | Feature engineering exploration |
| `notebooks/model_testing.ipynb` | Manual model testing (no formal test runner) |

## Known issues / Future work

- **Attention classification head is effectively a no-op at inference.** The `AdditiveAttention` layer operates over a sequence of length 1, so it numerically reduces to an identity — the `attention` head only adds `LayerNorm` + `Dropout(0.2)` over the `standard` head. Verified on the deployed models. Either remove it or reshape to attend over the window sequence.
- **Gap analysis** (`src/utils/gap_analysis.py`) compares training vs real-time feature distributions; results from live sessions are a work in progress.
- **Lab attack scripts** — `experiment/scripts/malicious/brute_force_ssh.sh` and `dos_hping.sh` are templates that send no traffic yet; there is no SSH-benign script (run manually).

## Documentation

- `AGENTS.md` — agent/development instructions (key commands, architecture, paths)
- [`experiment/README.md`](experiment/README.md) — lab environment and evaluation
- [`experiment/LAB_REPORT.local.md`](experiment/LAB_REPORT.local.md) — lab VM environment and per-session record template
