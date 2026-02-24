import pandas as pd
import glob
import json
from pathlib import Path

def assemble_cic_dataset(data_folder:str='../../data/raw/CIC-IDS2017', output_folder:str='../../data/processed/CIC-IDS2017'):
    """
    Assemble CIC-IDS2017 parquet files into a single dataframe
    with attack type labels derived from filenames
    """
    data_folder = Path(data_folder)
    all_files = list(data_folder.glob('*.parquet'))
    
    print(f"Found {len(all_files)} CIC-IDS2017 files:")
    dataframes = []
    
    for file_path in all_files:
        # Extract attack type from filename
        filename = file_path.stem  # Gets filename without extension
        attack_type = filename.replace('-no-metadata', '').split('-')[-1]
        
        # Handle special cases
        if 'Monday' in filename:
            attack_type = 'BENIGN'
        elif 'Tuesday' in filename:
            attack_type = 'Bruteforce'
        elif 'Wednesday' in filename:
            attack_type = 'DoS'
        elif 'Thursday' in filename:
            if 'Infiltration' in filename:
                attack_type = 'Infiltration'
            else:
                attack_type = 'WebAttacks'
        elif 'Friday' in filename:
            if 'DDoS' in filename:
                attack_type = 'DDoS'
            elif 'Portscan' in filename:
                attack_type = 'Portscan'
            elif 'Botnet' in filename:
                attack_type = 'Botnet'
        
        # Read parquet file
        print(f"Loading {filename}... (Attack: {attack_type})")
        df = pd.read_parquet(file_path)
        
        # Add attack type column
        df['attack_type'] = attack_type
        df['attack_label'] = 0 if attack_type == 'BENIGN' else 1
        
        # Add source file tracking (useful for debugging)
        df['source_file'] = filename
        
        dataframes.append(df)
    
    # Combine all dataframes
    print("\nCombining all files...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  - Total samples: {len(combined_df):,}")
    print(f"  - Total features: {len(combined_df.columns)}")
    print(f"  - Attack types: {combined_df['attack_type'].unique()}")
    
    # Show distribution
    print("\nClass distribution:")
    dist = combined_df['attack_type'].value_counts()
    for attack, count in dist.items():
        print(f"  - {attack}: {count:,} ({count/len(combined_df)*100:.1f}%)")

    output_file = Path(output_folder) / 'cic_assembled.parquet'
    combined_df.to_parquet(output_file, index=False)
    print(f"\nSaved assembled dataset to: {output_file}")

    metadata = {
        'total_samples': len(combined_df),
        'total_features': len(combined_df.columns),
        'class_distribution': dist.to_dict(),
        'attack_types': list(combined_df['attack_type'].unique()),
        'source_files': [f.name for f in all_files]
    }
    
    metadata_file = Path(output_folder) / 'cic_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to: {metadata_file}")
    
    return combined_df

if __name__ == "__main__":
    assemble_cic_dataset()
