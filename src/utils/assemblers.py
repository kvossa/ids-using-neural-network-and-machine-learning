import pandas as pd
import glob
from pathlib import Path

def assemble_cic_dataset(data_folder='../data/raw/CIC-IDS2017'):
    """
    Assemble CIC-IDS2017 parquet files into a single dataframe
    with attack type labels derived from filenames
    """
    data_folder = Path(data_folder)
    all_files = list(data_folder.glob('*.parquet'))
    
    print(f"📂 Found {len(all_files)} CIC-IDS2017 files:")
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
        print(f"  📖 Loading {filename}... (Attack: {attack_type})")
        df = pd.read_parquet(file_path)
        
        # Add attack type column
        df['attack_type'] = attack_type
        df['attack_label'] = 0 if attack_type == 'BENIGN' else 1
        
        # Add source file tracking (useful for debugging)
        df['source_file'] = filename
        
        dataframes.append(df)
    
    # Combine all dataframes
    print("\n🔄 Combining all files...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\n✅ Combined dataset:")
    print(f"  - Total samples: {len(combined_df):,}")
    print(f"  - Total features: {len(combined_df.columns)}")
    print(f"  - Attack types: {combined_df['attack_type'].unique()}")
    
    # Show distribution
    print("\n📊 Class distribution:")
    dist = combined_df['attack_type'].value_counts()
    for attack, count in dist.items():
        print(f"  - {attack}: {count:,} ({count/len(combined_df)*100:.1f}%)")
    
    return combined_df

def assemble_cic_dataset_memory_safe(data_folder='../data/raw/CIC-IDS2017', 
                                     chunksize=100000,
                                     sample_rate=1.0):
    """
    Memory-safe assembly with optional sampling for large datasets
    """
    data_folder = Path(data_folder)
    all_files = sorted(data_folder.glob('*.parquet'))
    
    print(f"📂 Found {len(all_files)} files")
    
    # First, get column structure from first file
    sample_df = pd.read_parquet(all_files[0])
    all_columns = sample_df.columns.tolist()
    
    # Initialize empty list for processed chunks
    processed_chunks = []
    
    for file_path in all_files:
        filename = file_path.stem
        attack_type = extract_attack_type(filename)
        
        print(f"📖 Processing {filename}...")
        
        # Read in chunks if file is large
        parquet_file = pd.read_parquet(file_path)
        
        # Optional sampling
        if sample_rate < 1.0:
            parquet_file = parquet_file.sample(frac=sample_rate, random_state=42)
        
        # Add labels
        parquet_file['attack_type'] = attack_type
        parquet_file['attack_label'] = (attack_type != 'BENIGN').astype(int)
        parquet_file['source_file'] = filename
        
        # Ensure same column order
        parquet_file = parquet_file[[col for col in all_columns if col in parquet_file.columns] 
                                     + ['attack_type', 'attack_label', 'source_file']]
        
        processed_chunks.append(parquet_file)
        
        # Optional: Clear memory periodically
        if len(processed_chunks) >= 3:
            print(f"  💾 Memory checkpoint: {len(processed_chunks)} files loaded")
    
    # Combine all
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    return final_df

def extract_attack_type(filename):
    """Extract attack type from CIC filename"""
    clean_name = filename.replace('-no-metadata', '')
    
    if 'Monday' in clean_name:
        return 'BENIGN'
    elif 'Tuesday' in clean_name:
        return 'Bruteforce'
    elif 'Wednesday' in clean_name:
        return 'DoS'
    elif 'Thursday' in clean_name:
        return 'WebAttacks' if 'Web' in clean_name else 'Infiltration'
    elif 'Friday' in clean_name:
        if 'DDoS' in clean_name:
            return 'DDoS'
        elif 'Portscan' in clean_name:
            return 'Portscan'
        elif 'Botnet' in clean_name:
            return 'Botnet'
    return 'Unknown'