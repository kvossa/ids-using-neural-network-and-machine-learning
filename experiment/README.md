# `experiment/` — IDS lab testing

Traffic orchestration, windowed capture, flow extraction (e.g. CICFlowMeter), live and batch inference aligned with `src/preprocessing`, and evaluation against temporal **ground truth**.

## Requirements

```bash
pip install -r experiment/requirements.txt
```

Plus: `tcpdump` (root or `CAP_NET_RAW`), `java` if using CICFlowMeter, optional attack tooling (`nmap`, `curl`, `dig`, `hydra`), and the project's inference dependencies (`tensorflow`/`keras`, `joblib`) aligned with training.

## Configuration

1. Copy [`config/lab.example.json`](config/lab.example.json) to `config/lab.json` and edit IPs, capture interface, and artifact paths.
2. (Optional) Prefer YAML? Copy [`config/lab.example.yaml`](config/lab.example.yaml) to `config/lab.yaml` and install `pyyaml`; [`config_loader.py`](config_loader.py) accepts both formats.
3. Fill in [`FICHA_LABORATORIO.md`](FICHA_LABORATORIO.md) (or a local, unversioned copy).

`config/lab.json`, `config/lab.yaml`, and `FICHA_LABORATORIO.local.md` are gitignored — keep internal IPs out of the repo.

## Models and datasets

Artifacts in `lab.json` (`preprocessing_pkl`, `model_keras`) must match the configured dataset:

- **CIC** (`--mode cic`) — two-stage: stage 1 is a binary Normal vs Attack gate (with calibrated threshold `threshold.json`); stage 2 maps flagged windows to Flood/Rare groups. Streaming output includes `stage1_result` / `stage1_confidence` per window.
- **UNSW** (`--mode unsw`) — single-stage multiclass prediction.

Inference classes live in `src/inference/` (`cic.py`, `unsw.py`, `cic_stream.py`, `unsw_stream.py`) and load default artifact paths from `src/config.py`; fine-tuned heads are supported via `--model-dir` (see below).

## Live / streaming inference (primary flow)

Capture from an interface, extract flows, and write predictions as they stream:

```bash
python experiment/scripts/infer_live.py --mode {cic,unsw} [--interface eth0] \
  [--output results/predictions.csv] [--model-dir models/classification/fine_tuned/cic] \
  [--log-features results/features.csv]
```

- `--mode` is required. `--interface` auto-detects if omitted.
- `--model-dir` switches to fine-tuned heads (`stage1.keras`/`stage2.keras` for CIC, `single_stage.keras` for UNSW).
- `--log-features` saves preprocessed feature vectors for gap analysis.
- Output CSV columns: `prediction,confidence,group,stage1_result,stage1_confidence,_timestamp`.

Run with `sudo` when the interface requires root (the script re-execs with the user's Python if the venv is lost).

## Batch inference

Flows CSV → preprocessing → windows → model:

```bash
python experiment/scripts/infer_batch.py --config experiment/config/lab.json \
  --flows-csv experiment/results/flows/session_flows.csv [--dataset CIC|UNSW]
```

- Windows are built with `step=1`; `window_t_start`/`window_t_end` are the first/last flow timestamps in each window (requires a `Timestamp` column for CIC).
- Output CSV columns: `window_index,window_t_start,window_t_end,prediction,confidence,group,stage1_result,stage1_confidence,source_flows_csv`.

## Evaluation

Temporal overlap of ground truth intervals vs prediction windows:

```bash
python experiment/scripts/evaluate_lab.py \
  --ground-truth experiment/results/ground_truth.csv \
  --predictions experiment/results/predictions.csv \
  --output experiment/results/lab_metrics.json
```

- Binary metrics (F1/precision/recall, benign-only FP rate, attack latency, mixed windows) by default.
- Group-level metrics with `--evaluate-groups --dataset {CIC,UNSW}` (GT labels mapped to Flood/Rare or UNSW confusion groups).
- Accepts both the batch format (`window_t_start`/`window_t_end`) and the live format (`_timestamp` as an instantaneous window).

## Orchestrator (optional automation)

[`scripts/orchestrator.py`](scripts/orchestrator.py) records ground-truth intervals and launches traffic scenarios:

```bash
sudo python experiment/scripts/orchestrator.py --config experiment/config/lab.json
```

Capture can be invoked per scenario or run manually via [`capture/window_capture.py`](capture/window_capture.py). If CICFlowMeter is disabled, feed the flows CSV to inference manually.

## Structure

| Path | Purpose |
|------|---------|
| `config/` | `lab.json` / `lab.yaml` (local, gitignored), `lab.example.json` / `lab.example.yaml` templates |
| `config_loader.py` | Loads JSON or YAML lab config |
| `scripts/normal/` | Reproducible benign traffic (`http_benign.sh`, `dns_loop.sh`) |
| `scripts/malicious/` | Attack scenarios (`port_scan.sh`, `brute_force_ssh.sh`, `dos_hping.sh`) |
| `scripts/orchestrator.py` | Coordination + `ground_truth.csv` |
| `scripts/infer_live.py` | Live/streaming inference |
| `scripts/infer_batch.py` | Batch inference (flows CSV → windows → model) |
| `scripts/evaluate_lab.py` | F1, benign FP, latency, mixed windows, group metrics |
| `capture/` | Windowed capture (`window_capture.py`) |
| `results/` | Outputs; only `results/templates/` is versioned (rest gitignored) |

## Ground truth and predictions

- GT format: headers in `results/templates/ground_truth.example.csv`.
- Predictions: `results/templates/predictions.example.csv` (batch format; live output adds `_timestamp`).

## Dataset notes

The `preprocessing.pkl` and model must correspond to the same dataset (`CIC` or `UNSW`) configured in `lab.json`. Lab traffic via CICFlowMeter aligns naturally with **CIC** mode (CIC-IDS2017-style columns).

For **UNSW-NB15**:

- Use `tcpdump` or `tshark` for windowed PCAP capture.
- Do **not** use CICFlowMeter as the main extractor when inferring with UNSW artifacts — its columns do not match the UNSW schema.
- Extract features with an UNSW-compatible flow (e.g. Argus + Zeek/Bro, or a custom extractor reproducing UNSW variables like `dur`, `sbytes`, `dbytes`, `sttl`, `ct_*`).
- Verify the final CSV has the columns expected by the UNSW pipeline before running `infer_batch.py`.
