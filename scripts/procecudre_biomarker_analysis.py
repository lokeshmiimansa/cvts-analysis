import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import warnings

# --- Configuration ---
# Update these paths to your actual file locations
REFERENCE_CSV = '/mnt/mdrive/home1/akshat/cvts-analysis/data/analysis/biomarker_reference.csv'
MEASUREMENTS_CSV = '/mnt/mdrive/home1/akshat/cvts-analysis/data/analysis/measurements.csv'

# Optional: If you create this file later, uncomment the line below and add the path
PROCEDURE_REFERENCE_CSV = None # e.g., '/mnt/mdrive/home1/akshat/CVT/data/analysis/procedure_reference.csv'

# Output settings
OUTPUT_DIR = 'biomarker_plots'
BATCH_SIZE = 25  # Patients per plot

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Robust CSV loader that attempts different delimiters.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"Loading {file_path}...")
    try:
        # Try standard CSV first
        df = pd.read_csv(file_path)
    except:
        try:
            # Try tab-separated
            df = pd.read_csv(file_path, sep='\t')
        except:
            # Try auto-detection engine
            df = pd.read_csv(file_path, sep=None, engine='python')
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(measurements, biomarkers, procedures=None):
    """
    Merges data, converts types, and calculates time differences.
    """
    print("Preprocessing data...")
    
    # 1. Ensure ID columns are strings to prevent merge errors
    measurements['measurement_concept_id'] = measurements['measurement_concept_id'].astype(str)
    measurements['person_id'] = measurements['person_id'].astype(str)
    measurements['procedure_concept_id'] = measurements['procedure_concept_id'].astype(str)
    
    biomarkers['measurement_concept_id'] = biomarkers['measurement_concept_id'].astype(str)
    
    # 2. Merge Measurements with Biomarker Reference
    # We use 'left' join to keep measurements even if biomarker info is missing
    merged = measurements.merge(
        biomarkers, 
        on='measurement_concept_id', 
        how='left',
        suffixes=('', '_ref')
    )
    
    # 3. Handle Procedure Names
    if procedures is not None:
        procedures['procedure_concept_id'] = procedures['procedure_concept_id'].astype(str)
        merged = merged.merge(procedures, on='procedure_concept_id', how='left')
        # Fill missing names with the ID
        merged['procedure_name'] = merged['procedure_name'].fillna('Procedure_' + merged['procedure_concept_id'])
    else:
        # No reference file provided, use ID as name
        merged['procedure_name'] = 'Procedure_' + merged['procedure_concept_id']

    # 4. Handle Biomarker Names (Fill missing with ID)
    if 'measurement_concept_name' not in merged.columns:
        merged['measurement_concept_name'] = 'Biomarker_' + merged['measurement_concept_id']
    else:
        merged['measurement_concept_name'] = merged['measurement_concept_name'].fillna('Biomarker_' + merged['measurement_concept_id'])

    # 5. Convert Datetimes
    # Using errors='coerce' turns bad dates into NaT (Not a Time)
    merged['procedure_datetime'] = pd.to_datetime(merged['procedure_datetime'], errors='coerce', utc=True)
    merged['measurement_datetime'] = pd.to_datetime(merged['measurement_datetime'], errors='coerce', utc=True)
    
    # Remove timezones for arithmetic
    merged['procedure_datetime'] = merged['procedure_datetime'].dt.tz_localize(None)
    merged['measurement_datetime'] = merged['measurement_datetime'].dt.tz_localize(None)
    
    # 6. Calculate Time Difference (days)
    # Drop rows where dates are missing
    merged = merged.dropna(subset=['procedure_datetime', 'measurement_datetime'])
    merged['t'] = (merged['measurement_datetime'] - merged['procedure_datetime']).dt.total_seconds() / (24 * 3600)
    
    return merged

def get_reference_values(df_subset):
    """
    Extracts range_low, range_high, and unit from a subset of data.
    """
    low, high, unit = None, None, None
    
    # Helper to clean and get first valid value
    def get_first_valid(series):
        valid = series.replace(['NULL', 'null', 'nan', 'NaN'], np.nan).dropna()
        if not valid.empty:
            return valid.iloc[0]
        return None

    if 'range_low' in df_subset.columns:
        val = get_first_valid(df_subset['range_low'])
        if val is not None:
            try: low = float(val)
            except: pass
            
    if 'range_high' in df_subset.columns:
        val = get_first_valid(df_subset['range_high'])
        if val is not None:
            try: high = float(val)
            except: pass
            
    if 'unit_name' in df_subset.columns:
        unit = get_first_valid(df_subset['unit_name'])
        
    return low, high, unit

def plot_trajectory_batch(df_batch, procedure_name, biomarker_name, ranges, batch_num, total_batches, save_folder):
    """
    Creates and saves a single plot.
    """
    low, high, unit = ranges
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot trajectories
    sns.lineplot(
        data=df_batch,
        x='t',
        y='value_as_number',
        hue='person_id',
        palette='tab10',
        marker='o',
        linewidth=2,
        alpha=0.8
    )
    
    # Add Reference Elements
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Procedure Date')
    
    if low is not None and high is not None:
        plt.axhline(y=low, color='orange', linestyle=':', linewidth=2, label=f'Low ({low})')
        plt.axhline(y=high, color='orange', linestyle=':', linewidth=2, label=f'High ({high})')
        plt.fill_between([df_batch['t'].min(), df_batch['t'].max()], low, high, color='green', alpha=0.05)
        
    # Annotations (Pre/Post arrows)
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    arrow_y = y_min + (y_range * 0.05)
    
    if df_batch['t'].min() < -1:
        plt.text(df_batch['t'].min()/2, arrow_y, 'Pre-Procedure', color='green', ha='center')
    if df_batch['t'].max() > 1:
        plt.text(df_batch['t'].max()/2, arrow_y, 'Post-Procedure', color='purple', ha='center')

    # Labels and Title
    y_label = f'{biomarker_name}' + (f' ({unit})' if unit else '')
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel('Days from Procedure', fontsize=12)
    plt.title(f'{biomarker_name} vs Time - {procedure_name}\n(Batch {batch_num}/{total_batches})', fontsize=14)
    
    # Legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Patient ID")
    plt.tight_layout()
    
    # Save
    safe_bio = "".join([c if c.isalnum() else "_" for c in biomarker_name])
    filename = f"{safe_bio}_batch_{batch_num}.png"
    plt.savefig(os.path.join(save_folder, filename), dpi=100)
    plt.close()
    print(f"    Saved: {filename}")

def main():
    print("--- Starting Biomarker Analysis ---")
    
    # 1. Load Data
    try:
        measurements_df = load_data(MEASUREMENTS_CSV)
        reference_df = load_data(REFERENCE_CSV)
        
        procedure_df = None
        if PROCEDURE_REFERENCE_CSV and os.path.exists(PROCEDURE_REFERENCE_CSV):
            procedure_df = load_data(PROCEDURE_REFERENCE_CSV)
            print("Procedure reference loaded.")
        else:
            print("No procedure reference found. Using concept IDs.")
            
    except Exception as e:
        print(f"CRITICAL ERROR LOADING DATA: {e}")
        return

    # 2. Process Data
    full_df = preprocess_data(measurements_df, reference_df, procedure_df)
    
    print(f"Total processed records: {len(full_df)}")
    
    # 3. Iterate and Plot
    unique_procedures = full_df['procedure_name'].unique()
    
    for proc in unique_procedures:
        print(f"\nProcessing Procedure: {proc}")
        proc_df = full_df[full_df['procedure_name'] == proc]
        
        unique_biomarkers = proc_df['measurement_concept_name'].unique()
        
        for bio in unique_biomarkers:
            bio_df = proc_df[proc_df['measurement_concept_name'] == bio]
            
            if bio_df.empty:
                continue
                
            # Create Folder Structure: Output / Procedure / Biomarker
            safe_proc = "".join([c if c.isalnum() else "_" for c in proc])
            safe_bio = "".join([c if c.isalnum() else "_" for c in bio])
            
            save_path = os.path.join(OUTPUT_DIR, safe_proc, safe_bio)
            os.makedirs(save_path, exist_ok=True)
            
            # Get ranges
            ranges = get_reference_values(bio_df)
            
            # Batching
            patients = bio_df['person_id'].unique()
            num_batches = math.ceil(len(patients) / BATCH_SIZE)
            
            print(f"  Biomarker: {bio} ({len(patients)} patients -> {num_batches} batches)")
            
            for i in range(num_batches):
                batch_patients = patients[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                batch_df = bio_df[bio_df['person_id'].isin(batch_patients)]
                
                plot_trajectory_batch(
                    batch_df, 
                    proc, 
                    bio, 
                    ranges, 
                    i+1, 
                    num_batches, 
                    save_path
                )

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()




