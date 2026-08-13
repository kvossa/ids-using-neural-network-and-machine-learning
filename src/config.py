"""Centralized paths for preprocessing artifacts, models, and data."""

PREPROC_PATHS = {
    "cic": {
        "binary_preprocessor": "models/preprocessing/binary/cic/preprocessing.pkl",
        "binary_encoder": "models/preprocessing/binary/cic/label_encoder.pkl",
        "multiclass_encoder": "models/preprocessing/multiclass/cic/label_encoder.pkl",
        "feature_names": "models/preprocessing/binary/cic/feature_names.npy",
    },
    "unsw": {
        "binary_preprocessor": "models/preprocessing/binary/unsw/preprocessing.pkl",
        "binary_encoder": "models/preprocessing/binary/unsw/label_encoder.pkl",
        "multiclass_encoder": "models/preprocessing/multiclass/unsw/label_encoder.pkl",
        "feature_names": "models/preprocessing/binary/unsw/feature_names.npy",
    },
}

CIC_STAGE1_DIR = "models/classification/two_stage/cic/stage1"
CIC_STAGE1 = f"{CIC_STAGE1_DIR}/best_model_binary.keras"
CIC_STAGE1_THRESHOLD = f"{CIC_STAGE1_DIR}/threshold.json"

CIC_STAGE2_DIR = "models/classification/two_stage/cic/stage2/bruterare"
CIC_STAGE2 = f"{CIC_STAGE2_DIR}/best_model_multiclass.keras"

UNSW_CONFUSION_GROUPS_DIR = "models/classification/single_stage/unsw/confusion_groups"
UNSW_MODEL = f"{UNSW_CONFUSION_GROUPS_DIR}/baseline/best_model_multiclass.keras"

FINE_TUNED_DIR = "models/classification/fine_tuned"
FINE_TUNED = {
    "cic_stage1": f"{FINE_TUNED_DIR}/cic/stage1.keras",
    "cic_stage2": f"{FINE_TUNED_DIR}/cic/stage2.keras",
    "unsw": f"{FINE_TUNED_DIR}/unsw/single_stage.keras",
}

FINETUNE_DATA_DIR = "finetune_data"

DATA_PATHS = {
    "cic": {
        "train": "data/processed/CIC-IDS2017/splits/train/data.parquet",
        "test": "data/processed/CIC-IDS2017/splits/test/data.parquet",
        "val": "data/processed/CIC-IDS2017/splits/val/data.parquet",
    },
    "unsw": {
        "train": "data/processed/UNSW-NB15/splits/train.csv",
        "test": "data/processed/UNSW-NB15/splits/test.csv",
        "val": "data/processed/UNSW-NB15/splits/validation.csv",
    },
}

REPORT_PATHS = {
    "cic_stage1": "reports/metrics/cic/results/two_stage/stage1",
    "cic_stage2": "reports/metrics/cic/results/bruterare",
    "unsw_confusion_groups": "reports/metrics/unsw/results/single_stage/confusion_groups",
}
