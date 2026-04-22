import joblib
import numpy as np
import pandas as pd
from pathlib import Path

for DATASET, label_col, normal_label, drop_cols, train_path, loader in [
    ("UNSW", "attack_cat", "Normal",
     ["attack_cat", "label"],
     Path("data/processed/UNSW-NB15/splits/train.csv"), "csv"),
    ("CIC",  "attack_type", "BENIGN",
     ["Label", "attack_label", "attack_type", "source_file"],
     Path("data/processed/CIC-IDS2017/splits/train/data.parquet"), "parquet"),
]:
    print(f"\n{'='*60}")
    print(f"  SEPARABILIDAD POR CLASE — {DATASET}")
    print(f"{'='*60}")

    train_df = pd.read_csv(train_path) if loader == "csv" else pd.read_parquet(train_path)
    
    preprocessor  = joblib.load(f"models/preprocessing/multiclass/{DATASET.lower()}/preprocessing.pkl")
    label_encoder = joblib.load(f"models/preprocessing/multiclass/{DATASET.lower()}/label_encoder.pkl")

    y_raw  = train_df[label_col]
    X      = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_proc = preprocessor.transform(X)
    y_enc  = label_encoder.transform(y_raw)

    feature_names = preprocessor.pipeline.named_steps['feature_selection'].selected_features_
    X_arr = X_proc.values if hasattr(X_proc, 'values') else X_proc
    
    # Separación de cada clase vs todas las demás
    classes     = label_encoder.classes_
    X_normal    = X_arr[y_enc == label_encoder.transform([normal_label])[0]]
    
    print(f"\n{'Clase':<20} {'n':>8} {'Sep vs Normal':>15} {'Sep máx feat':>14}")
    print("-" * 60)
    
    class_separations = {}
    for cls_name in classes:
        cls_idx = label_encoder.transform([cls_name])[0]
        X_cls   = X_arr[y_enc == cls_idx]
        
        if len(X_cls) == 0:
            continue
        
        sep_vs_normal = np.abs(
            X_normal.mean(axis=0) - X_cls.mean(axis=0)
        ).max()
        
        class_separations[cls_name] = sep_vs_normal
        print(f"{cls_name:<20} {len(X_cls):>8,} {sep_vs_normal:>15.4f}")
    
    # Recomendación de window_size
    avg_sep = np.mean(list(class_separations.values()))
    print(f"\n  Separación promedio : {avg_sep:.4f}")
    print(f"  Clases con sep > 0.1: {sum(1 for v in class_separations.values() if v > 0.1)}")
    print(f"  Clases con sep > 0.3: {sum(1 for v in class_separations.values() if v > 0.3)}")
    
    if avg_sep > 0.2:
        print(f"\n  → Separabilidad ALTA: window_size=10 recomendado")
    elif avg_sep > 0.05:
        print(f"\n  → Separabilidad MEDIA: window_size=5 recomendado, validar con test")
    else:
        print(f"\n  → Separabilidad BAJA: window_size=1, problema en features")


        