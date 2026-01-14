import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

# --- Configuration ---
# File paths
REFERENCE_CSV = '/mnt/mdrive/home1/akshat/cvts-analysis/data/analysis/biomarker_reference.csv'  
MEASUREMENTS_CSV = '/mnt/mdrive/home1/akshat/cvts-analysis/data/analysis/measurements.csv'  

# Plotting configuration
batch_size = 25  # Number of patients per plot
base_output_folder = 'biomarker_analysis'

# --- Load Data ---
print("Loading data...")

# Try to detect the delimiter automatically
try:
    reference_df = pd.read_csv(REFERENCE_CSV, sep=None, engine='python')
except:
    # If that fails, try common delimiters
    try:
        reference_df = pd.read_csv(REFERENCE_CSV, sep='\t')
    except:
        reference_df = pd.read_csv(REFERENCE_CSV)

measurements_df = pd.read_csv(MEASUREMENTS_CSV)

# Strip whitespace from column names
reference_df.columns = reference_df.columns.str.strip()
measurements_df.columns = measurements_df.columns.str.strip()

print(f"Reference biomarkers loaded: {len(reference_df)}")
print(f"Measurements loaded: {len(measurements_df)}")

# Debug: Print column names
print(f"\nReference CSV columns: {list(reference_df.columns)}")
print(f"Reference CSV shape: {reference_df.shape}")
print(f"\nFirst few rows of reference:")
print(reference_df.head())
print(f"\nMeasurements CSV columns: {list(measurements_df.columns)}")

# Check if required columns exist in reference_df
required_cols = ['measurement_concept_id', 'measurement_concept_name', 'range_low', 'range_high', 'unit_name']
missing_cols = [col for col in required_cols if col not in reference_df.columns]
if missing_cols:
    print(f"\nWARNING: Missing columns in reference CSV: {missing_cols}")
    print("Available columns:", list(reference_df.columns))

# --- Merge to get biomarker names, ranges, and units ---
# Use LEFT join to keep all measurements, even if not in reference
cols_to_merge = ['measurement_concept_id']
if 'measurement_concept_name' in reference_df.columns:
    cols_to_merge.append('measurement_concept_name')
if 'range_low' in reference_df.columns:
    cols_to_merge.append('range_low')
if 'range_high' in reference_df.columns:
    cols_to_merge.append('range_high')
if 'unit_name' in reference_df.columns:
    cols_to_merge.append('unit_name')

# Use suffixes to identify which columns come from which dataframe
merged_df = measurements_df.merge(
    reference_df[cols_to_merge],
    on='measurement_concept_id',
    how='left',
    suffixes=('_meas', '_ref')
)

# Rename columns for clarity - prefer reference values over measurement values
if 'range_low_ref' in merged_df.columns:
    merged_df['range_low'] = merged_df['range_low_ref']
elif 'range_low_meas' in merged_df.columns:
    merged_df['range_low'] = merged_df['range_low_meas']

if 'range_high_ref' in merged_df.columns:
    merged_df['range_high'] = merged_df['range_high_ref']
elif 'range_high_meas' in merged_df.columns:
    merged_df['range_high'] = merged_df['range_high_meas']

# For biomarkers not in reference, use measurement_concept_id as name
if 'measurement_concept_name' not in merged_df.columns:
    merged_df['measurement_concept_name'] = merged_df['measurement_concept_id'].astype(str)
else:
    # Fill missing names with concept_id
    merged_df['measurement_concept_name'] = merged_df['measurement_concept_name'].fillna(
        'Biomarker_' + merged_df['measurement_concept_id'].astype(str)
    )

print(f"\nMerged columns: {list(merged_df.columns)}")

# --- Process datetime columns ---
merged_df['procedure_datetime'] = pd.to_datetime(merged_df['procedure_datetime'], format='mixed', errors='coerce', utc=True)
merged_df['measurement_datetime'] = pd.to_datetime(merged_df['measurement_datetime'], format='mixed', errors='coerce', utc=True)

# Remove timezone information
merged_df['procedure_datetime'] = merged_df['procedure_datetime'].dt.tz_localize(None)
merged_df['measurement_datetime'] = merged_df['measurement_datetime'].dt.tz_localize(None)

# Calculate time difference in days
merged_df['t'] = (merged_df['measurement_datetime'] - merged_df['procedure_datetime']).dt.days

# Convert person_id to string
merged_df['person_id'] = merged_df['person_id'].astype(str)

print("\nDatetime processing complete.")
print(f"Total records after merge: {len(merged_df)}")

# Show all unique measurement_concept_ids in the data
unique_concept_ids = merged_df['measurement_concept_id'].unique()
print(f"\nTotal unique measurement_concept_ids in measurements: {len(unique_concept_ids)}")

# --- Group by Biomarker and Generate Plots ---
# Use measurement_concept_name which now includes all biomarkers
unique_biomarkers = merged_df['measurement_concept_name'].unique()
print(f"\nGenerating plots for {len(unique_biomarkers)} biomarkers...")

for biomarker_name in unique_biomarkers:
    print(f"\n{'='*60}")
    print(f"Processing: {biomarker_name}")
    print(f"{'='*60}")
    
    # Filter data for this biomarker
    biomarker_df = merged_df[merged_df['measurement_concept_name'] == biomarker_name].copy()
    
    if len(biomarker_df) == 0:
        print(f"No data found for {biomarker_name}, skipping...")
        continue
    
    # Get range values and unit (use first non-null value)
    low_threshold = None
    high_threshold = None
    unit_name = None
    
    # Debug: Check what values are present
    if 'range_low' in biomarker_df.columns:
        print(f"DEBUG - range_low unique values: {biomarker_df['range_low'].unique()[:5]}")
        print(f"DEBUG - range_low non-null count: {biomarker_df['range_low'].notna().sum()}")
    
    if 'range_high' in biomarker_df.columns:
        print(f"DEBUG - range_high unique values: {biomarker_df['range_high'].unique()[:5]}")
        print(f"DEBUG - range_high non-null count: {biomarker_df['range_high'].notna().sum()}")
    
    # Check if columns exist before accessing
    if 'range_low' in biomarker_df.columns:
        # Filter out string 'NULL' values and actual NaN
        valid_low = biomarker_df['range_low'].replace('NULL', np.nan).replace('null', np.nan).dropna()
        if not valid_low.empty:
            try:
                low_threshold = float(valid_low.iloc[0])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert range_low to float: {valid_low.iloc[0]}")
    
    if 'range_high' in biomarker_df.columns:
        # Filter out string 'NULL' values and actual NaN
        valid_high = biomarker_df['range_high'].replace('NULL', np.nan).replace('null', np.nan).dropna()
        if not valid_high.empty:
            try:
                high_threshold = float(valid_high.iloc[0])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert range_high to float: {valid_high.iloc[0]}")
    
    if 'unit_name' in biomarker_df.columns:
        valid_unit = biomarker_df['unit_name'].replace('NULL', np.nan).replace('null', np.nan).dropna()
        if not valid_unit.empty:
            unit_name = valid_unit.iloc[0]
    
    print(f"Low threshold: {low_threshold if low_threshold is not None else 'NULL'}")
    print(f"High threshold: {high_threshold if high_threshold is not None else 'NULL'}")
    print(f"Unit: {unit_name if unit_name is not None else 'NULL'}")
    
    # Create output folder for this biomarker
    # Clean the biomarker name for folder creation
    safe_biomarker_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in biomarker_name)
    output_folder = os.path.join(base_output_folder, safe_biomarker_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}/")
    
    # Setup batching
    unique_patients = biomarker_df['person_id'].unique()
    num_batches = math.ceil(len(unique_patients) / batch_size)
    
    print(f"Total patients: {len(unique_patients)}")
    print(f"Generating {num_batches} plots...")
    
    # Generate plots for each batch
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Get batch patients
        batch_patients = unique_patients[start_idx:end_idx]
        batch_df = biomarker_df[biomarker_df['person_id'].isin(batch_patients)]
        
        # --- Plotting ---
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        
        ax = sns.lineplot(
            data=batch_df,
            x='t',
            y='value_as_number',
            hue='person_id',
            palette='bright',
            marker='o',
            linewidth=1.5
        )
        
        # A. Reference Line (Procedure Date)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Procedure Date')
        
        # B. Threshold Lines (if available)
        if low_threshold is not None and high_threshold is not None:
            plt.axhline(y=low_threshold, color='orange', linestyle='--', linewidth=2, label=f'Low ({low_threshold})')
            plt.axhline(y=high_threshold, color='orange', linestyle='--', linewidth=2, label=f'High ({high_threshold})')
            plt.axhspan(low_threshold, high_threshold, color='green', alpha=0.05)
        
        # C. Phase Arrows
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        arrow_y_pos = y_min - (y_range * 0.12)
        t_min, t_max = batch_df['t'].min(), batch_df['t'].max()
        
        # Pre-Procedure Arrow
        if t_min < 0:
            ax.annotate('', xy=(t_min, arrow_y_pos), xytext=(-0.5, arrow_y_pos),
                       arrowprops=dict(arrowstyle="->", color='green', lw=2), annotation_clip=False)
            ax.text(t_min/2, arrow_y_pos, 'Pre-Procedure', ha='center', va='bottom', 
                   color='green', fontweight='bold')
        
        # Post-Procedure Arrow
        if t_max > 0:
            ax.annotate('', xy=(t_max, arrow_y_pos), xytext=(0.5, arrow_y_pos),
                       arrowprops=dict(arrowstyle="->", color='purple', lw=2), annotation_clip=False)
            ax.text(t_max/2, arrow_y_pos, 'Post-Procedure', ha='center', va='bottom', 
                   color='purple', fontweight='bold')
        
        # D. Formatting
        plt.xlabel('Days relative to Procedure (t)', labelpad=25)
        
        # Y-axis label with unit if available
        y_label = f'{biomarker_name} Measurement'
        if unit_name is not None:
            y_label += f' ({unit_name})'
        plt.ylabel(y_label)
        
        plt.title(f'{biomarker_name} Trajectories (Batch {i+1} of {num_batches})')
        
        # Legend settings
        plt.legend(title='Patient ID', bbox_to_anchor=(1.05, 1), loc='upper left', 
                  ncol=1, fontsize='small')
        plt.subplots_adjust(bottom=0.2, right=0.80)
        
        # E. Save and Close
        filename = f"{safe_biomarker_name}_batch_{i+1}.png"
        save_path = os.path.join(output_folder, filename)
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"  Saved {filename}")
        plt.close()

print("\n" + "="*60)
print("All biomarkers processed successfully!")
print("="*60)