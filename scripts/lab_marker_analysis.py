# ------------------------------------------------
# SCRIPT: Generate Interactive Lab Marker Plots by Procedure
# ------------------------------------------------
# This script reads lab marker measurement data, processes it,
# and generates interactive Plotly plots for each unique combination
# of procedure and measurement concept IDs.
# Each plot visualizes lab marker values over time relative to the procedure date,
# with lines colored by length of stay (LOS) and with post operative LOS and includes normal range shading.
# ------------------------------------------------
# IMPORTS
# ------------------------------------------------
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------
DATA_PATH = '../data/analysis/labmarker_measurement.csv' # Update this path if necessary
OUTPUT_DIR = '../data/plots'
# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ------------------------------------------------
# 1) LOAD & PREPROCESS DATA
# ------------------------------------------------
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
# Convert timestamps
df["measurement_datetime"] = pd.to_datetime(df["measurement_datetime"], format="mixed", errors="coerce")
df["procedure_datetime"] = pd.to_datetime(df["procedure_datetime"], format="mixed", errors="coerce")
# Calculate Day relative to procedure (t=0)
df["day_from_procedure"] = (
    df["measurement_datetime"].dt.normalize() - df["procedure_datetime"].dt.normalize()
).dt.days
# Rename value column for consistency
df = df.rename(columns={"value_as_number": "labmarker_value"})
# Drop rows essential for plotting
df = df.dropna(subset=["person_id", "day_from_procedure", "labmarker_value", "los_days"])
# Ensure ID columns are integers (removes .0 if present)
if 'procedure_concept_id' in df.columns:
    df['procedure_concept_id'] = df['procedure_concept_id'].astype(int)
if 'measurement_concept_id' in df.columns:
    df['measurement_concept_id'] = df['measurement_concept_id'].astype(int)
# ------------------------------------------------
# 2) DEFINE LISTS TO ITERATE
# ------------------------------------------------
# We get unique IDs directly from the data.
# You can also hardcode these lists if you only want specific ones generated.
unique_procedures = df['procedure_concept_id'].unique()
unique_measurements = df['measurement_concept_id'].unique()
print(f"Found {len(unique_procedures)} procedures and {len(unique_measurements)} measurements.")
# ------------------------------------------------
# 3) GENERATE PLOTS LOOP
# ------------------------------------------------
for proc_id in unique_procedures:
    for meas_id in unique_measurements:
        # Filter for the specific Procedure AND Measurement pair
        subset = df[
            (df['procedure_concept_id'] == proc_id) &
            (df['measurement_concept_id'] == meas_id)
        ].copy()
        # Skip if no data exists for this combination
        if subset.empty:
            print(f"Skipping Proc: {proc_id} | Meas: {meas_id} (no data)")
            continue
        print(f"Generating plot for Proc: {proc_id} | Meas: {meas_id} (n={len(subset)})")
        # --- Plot Logic (Adapted from your rough.py) ---
        # Color mapping based on LOS
        los_min = subset["los_days"].min()
        los_max = subset["los_days"].max()
        colorscale = px.colors.sequential.Plasma
        def los_to_color(los):
            if los_max == los_min:
                t = 0.5
            else:
                t = (los - los_min) / (los_max - los_min)
            return px.colors.sample_colorscale(colorscale, t)[0]
        fig = go.Figure()
        # Add traces for each patient
        for person_id, g in subset.sort_values("day_from_procedure").groupby("person_id"):
            los = float(g["los_days"].iloc[0])
            color = los_to_color(los)
            fig.add_trace(
                go.Scatter(
                    x=g["day_from_procedure"],
                    y=g["labmarker_value"],
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    customdata=np.column_stack([
                        np.repeat(person_id, len(g)),
                        np.repeat(los, len(g))
                    ]),
                    hovertemplate=(
                        "Patient: %{customdata[0]}<br>"
                        "LOS: %{customdata[1]} days<br>"
                        "Day: %{x}<br>"
                        "Value: %{y}<extra></extra>"
                    ),
                    showlegend=False
                )
            )
        # Add Procedure day line (t = 0)
        fig.add_vline(
            x=0,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text="Procedure day",
            annotation_position="top"
        )
        # Normal range shading (only if range is consistent in this subset)
        range_low = subset["range_low"].dropna().unique()
        range_high = subset["range_high"].dropna().unique()
        if len(range_low) == 1 and len(range_high) == 1:
            fig.add_hrect(
                y0=range_low[0],
                y1=range_high[0],
                fillcolor="green",
                opacity=0.12,
                line_width=0,
                annotation_text="Normal range",
                annotation_position="top left"
            )
        # LOS colorbar (dummy trace)
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    colorscale=colorscale,
                    cmin=los_min,
                    cmax=los_max,
                    color=[los_min, los_max],
                    showscale=True,
                    colorbar=dict(
                        title="LOS (days)",
                        len=0.85
                    ),
                    size=10,
                    opacity=0
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )
        # Update Layout
        fig.update_layout(
            title=f"Measurement {meas_id} for Procedure {proc_id}",
            xaxis_title="Days from procedure (t = 0)",
            yaxis_title="Lab Marker Value",
            template="simple_white",
            margin=dict(l=40, r=40, t=50, b=40)
            # Removed fixed width/height so it fills the iframe container
        )
        # ------------------------------------------------
        # 4) SAVE TO FILE
        # ------------------------------------------------
        # File name format: PROCEDUREID_MEASUREMENTID.html
        filename = f"{OUTPUT_DIR}/{proc_id}_{meas_id}.html"
        # write_html saves the interactive plot
        fig.write_html(filename, include_plotlyjs='cdn')
print("All plots generated successfully.")






















