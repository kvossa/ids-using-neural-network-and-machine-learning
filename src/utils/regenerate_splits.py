"""
regenerate_unsw_splits.py
Regenera splits limpios desde cero combinando los archivos oficiales
y eliminando duplicados antes de dividir.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

def regenerate_splits(
    train_file: str = 'data/raw/UNSW_NB15/UNSW_NB15_training-set.csv',
    test_file:  str = 'data/raw/UNSW_NB15/UNSW_NB15_testing-set.csv',
    output_dir: str = 'data/processed/UNSW-NB15/splits',
    test_size:  float = 0.20,
    val_size:   float = 0.10,
    random_state: int = 42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Combinar ambos archivos oficiales
    print("Cargando archivos oficiales...")
    df_train = pd.read_csv(train_file)
    df_test  = pd.read_csv(test_file)

    print(f"  Training-set oficial : {len(df_train):,}")
    print(f"  Testing-set oficial  : {len(df_test):,}")

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    print(f"  Total combinado      : {len(df_full):,}")

    # 2. Eliminar duplicados exactos
    feature_cols = [c for c in df_full.columns if c not in ['id']]
    df_full = df_full.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    print(f"  Tras deduplicación   : {len(df_full):,}")

    # 3. Verificar columna de etiqueta multiclase
    # UNSW tiene 'attack_cat' (multiclase) y 'label' (binaria)
    assert 'attack_cat' in df_full.columns, "No se encontró 'attack_cat'"

    # Limpiar espacios en attack_cat si los hay
    df_full['attack_cat'] = df_full['attack_cat'].str.strip()

    print("\nDistribución de clases (completa):")
    print(df_full['attack_cat'].value_counts())

    # 4. Split estratificado: train / temp
    train_df, temp_df = train_test_split(
        df_full,
        test_size=(test_size + val_size),
        stratify=df_full['attack_cat'],
        random_state=random_state,
    )

    # 5. Split temp → test / val
    relative_val = val_size / (test_size + val_size)
    test_df, val_df = train_test_split(
        temp_df,
        test_size=relative_val,
        stratify=temp_df['attack_cat'],
        random_state=random_state,
    )

    # 6. Verificar que no haya solapamiento
    feature_cols_check = [c for c in df_full.columns if c not in ['id', 'attack_cat', 'label']]
    train_test_overlap = train_df[feature_cols_check].merge(
        test_df[feature_cols_check], how='inner'
    )
    train_val_overlap = train_df[feature_cols_check].merge(
        val_df[feature_cols_check], how='inner'
    )
    print(f"\nVerificación de solapamiento:")
    print(f"  Train ∩ Test : {len(train_test_overlap):,}  ← debe ser 0")
    print(f"  Train ∩ Val  : {len(train_val_overlap):,}  ← debe ser 0")

    # 7. Resumen
    total = len(df_full)
    print(f"\nSplits generados:")
    print(f"  Train : {len(train_df):,}  ({len(train_df)/total*100:.1f}%)")
    print(f"  Test  : {len(test_df):,}   ({len(test_df)/total*100:.1f}%)")
    print(f"  Val   : {len(val_df):,}    ({len(val_df)/total*100:.1f}%)")

    print("\nDistribución train:")
    print(train_df['attack_cat'].value_counts(normalize=True).round(3))

    # 8. Guardar
    train_df.to_csv(output_path / 'train.csv',      index=False)
    test_df.to_csv( output_path / 'test.csv',       index=False)
    val_df.to_csv(  output_path / 'validation.csv', index=False)

    metadata = {
        'total_after_dedup': len(df_full),
        'train_samples': len(train_df),
        'test_samples':  len(test_df),
        'val_samples':   len(val_df),
        'test_size':     test_size,
        'val_size':      val_size,
        'random_state':  random_state,
        'overlap_train_test': len(train_test_overlap),
        'overlap_train_val':  len(train_val_overlap),
        'train_distribution': train_df['attack_cat'].value_counts().to_dict(),
        'test_distribution':  test_df['attack_cat'].value_counts().to_dict(),
        'val_distribution':   val_df['attack_cat'].value_counts().to_dict(),
    }
    with open(output_path / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSplits guardados en: {output_path}")
    return train_df, test_df, val_df


if __name__ == "__main__":
    regenerate_splits()