"""
create_unsw_validation.py
Creates validation set from UNSW-NB15 training data
Usage: python create_unsw_validation.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import argparse

def create_unsw_validation(train_file='../../data/raw/UNSW_NB15/UNSW_NB15_training-set.csv',
                          test_file='../../data/raw/UNSW_NB15/UNSW_NB15_testing-set.csv',
                          output_folder='../../data/processed/UNSW-NB15/splits',
                          val_size=0.15,
                          random_state=42):
    """
    Create validation set from UNSW training data
    """
    print("="*60)
    print("UNSW-NB15 Validation Set Creator")
    print("="*60)
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print(f"\nLoading training file: {train_file}")
    train_full = pd.read_csv(train_file)
    print(f"  - Training samples: {len(train_full):,}")
    
    # Load test data (will remain untouched)
    print(f"\nLoading test file: {test_file}")
    test_full = pd.read_csv(test_file)
    print(f"  - Test samples: {len(test_full):,}")
    
    # Check if label column exists (UNSW uses 'label' for binary: 0=normal, 1=attack)
    if 'label' not in train_full.columns:
        print("Warning: 'label' column not found. Please check column names.")
        print(f"Available columns: {train_full.columns.tolist()}")
        return None
    
    # Split training into train and validation
    print(f"\nSplitting training data (validation size: {val_size*100:.0f}%)...")
    train_data, val_data = train_test_split(
        train_full,
        test_size=val_size,
        stratify=train_full['label'],  # Maintain attack/normal ratio
        random_state=random_state
    )
    
    print(f"  - New training set: {len(train_data):,} samples")
    print(f"  - Validation set: {len(val_data):,} samples")
    print(f"  - Test set (unchanged): {len(test_full):,} samples")
    
    # Verify class distribution
    print("\nClass distribution check:")
    print("\n  Original training:")
    orig_dist = train_full['label'].value_counts(normalize=True) * 100
    print(f"    Normal (0): {orig_dist.get(0, 0):.1f}%, Attack (1): {orig_dist.get(1, 0):.1f}%")
    
    print("\n  New training:")
    train_dist = train_data['label'].value_counts(normalize=True) * 100
    print(f"    Normal (0): {train_dist.get(0, 0):.1f}%, Attack (1): {train_dist.get(1, 0):.1f}%")
    
    print("\n  Validation:")
    val_dist = val_data['label'].value_counts(normalize=True) * 100
    print(f"    Normal (0): {val_dist.get(0, 0):.1f}%, Attack (1): {val_dist.get(1, 0):.1f}%")
    
    print("\n  Test (unchanged):")
    test_dist = test_full['label'].value_counts(normalize=True) * 100
    print(f"    Normal (0): {test_dist.get(0, 0):.1f}%, Attack (1): {test_dist.get(1, 0):.1f}%")
    
    # Save splits
    print("\nSaving splits...")
    train_data.to_csv(output_path / 'train.csv', index=False)
    val_data.to_csv(output_path / 'validation.csv', index=False)
    test_full.to_csv(output_path / 'test.csv', index=False)
    
    # Save metadata
    metadata = {
        'original_training_samples': len(train_full),
        'original_test_samples': len(test_full),
        'new_train_samples': len(train_data),
        'validation_samples': len(val_data),
        'test_samples': len(test_full),
        'val_size': val_size,
        'random_state': random_state,
        'train_distribution': train_data['label'].value_counts().to_dict(),
        'val_distribution': val_data['label'].value_counts().to_dict(),
        'test_distribution': test_full['label'].value_counts().to_dict()
    }
    
    metadata_file = output_path / 'split_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nUNSW validation set creation complete!")
    print(f"Files saved to: {output_path}")
    print(f"  - train.parquet")
    print(f"  - validation.parquet")
    print(f"  - test.parquet")
    print(f"  - split_metadata.json")
    
    return train_data, val_data, test_full

if __name__ == "__main__":
    create_unsw_validation()