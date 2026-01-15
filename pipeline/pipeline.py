import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import textwrap
import plotly.graph_objects as go
import plotly.express as px
import os
from sqlalchemy import create_engine
from jinja2 import Environment, FileSystemLoader
import pipeline.config as config
import pipeline.queries as queries

# ==========================================
# 1. SETUP & DB CONNECTION
# ==========================================
def get_db_engine():
    db_url = f"postgresql+psycopg2://{config.DB_CONFIG['user']}:{config.DB_CONFIG['password']}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['dbname']}"
    return create_engine(db_url)

def setup_directories():
    for path in [config.OUTPUT_DIR, config.IMAGE_DIR, config.INTERACTIVE_PLOT_DIR, config.DATA_DIR]:
        os.makedirs(path, exist_ok=True)
    print("Directories created.")

# ==========================================
# 2. DATA EXTRACTION
# ==========================================
def fetch_data():
    engine = get_db_engine()
    procs = config.SELECTED_PROCEDURES
    meas = config.SELECTED_MEASUREMENTS

    print("Fetching Summary Stats (Table 3.1)...")
    df_summary = pd.read_sql(queries.get_summary_stats_query(procs), engine)
    
    print("Fetching Procedure Stats (Table 3.2)...")
    df_procedures = pd.read_sql(queries.get_procedure_stats_query(procs), engine)

    print("Fetching Lab Frequencies (Table 3.3)...")
    df_lab_freq = pd.read_sql(queries.get_lab_freq_stats_query(procs), engine)

    print("Fetching Mortality Stats (Table 5.1)...")
    df_mortality = pd.read_sql(queries.get_mortality_stats_query(procs), engine)
    
    print("Fetching Appendix Lab Stats...")
    df_lab_summary = pd.read_sql(queries.get_lab_summary_query(procs), engine)

    print("Fetching Master Cohort Data (For Plots)...")
    df_master = pd.read_sql(queries.get_master_data_query(procs), engine)
    
    print("Fetching Lab Measurement Data (For Interactive Plots)...")
    df_labs = pd.read_sql(queries.get_lab_measurement_query(procs, meas), engine)
    
    return {
        "summary": df_summary,
        "procedures": df_procedures,
        "lab_freq": df_lab_freq,
        "mortality": df_mortality,
        "lab_appendix": df_lab_summary,
        "master": df_master,
        "labs_interactive": df_labs
    }

# ==========================================
# 3. DATA CLEANING & PROCESSING
# ==========================================
def process_master_data(df):
    print("Cleaning Master Data...")
    
    # 1. Datetime Conversions
    df['procedure_date'] = pd.to_datetime(df['procedure_date'], errors='coerce')
    df['index_admit_date'] = pd.to_datetime(df['index_admit_date'], errors='coerce')
    
    # 2. Derive Death Datetime & Clean Invalid Deaths
    df['death_datetime'] = pd.NaT
    if 'is_death' in df.columns and 'days_proc_to_death' in df.columns:
        mask = (df['is_death'] == 1) & (df['days_proc_to_death'].notna())
        df.loc[mask, 'death_datetime'] = (
            df.loc[mask, 'procedure_date'] + 
            pd.to_timedelta(df.loc[mask, 'days_proc_to_death'], unit='D')
        )
    
    # Filter: Keep if alive OR (dead AND procedure <= death)
    clean_mask = (df['death_datetime'].isna()) | (df['procedure_date'] <= df['death_datetime'])
    dropped_count = len(df) - clean_mask.sum()
    if dropped_count > 0:
        print(f"  Dropped {dropped_count} records where procedure_date > death_date.")
        df = df[clean_mask].copy()

    # 3. Age Classes
    bins = [0, 2, 18, 40, 60, 150]
    labels_age = ['Infant', 'Pediatric', 'Young Adult', 'Adult', 'Senior']
    df['age_class'] = pd.cut(df['age_at_procedure'], bins=bins, labels=labels_age, right=False)
    
    # 4. Readmission Logic
    df['is_readmitted'] = df['readmission_date'].notnull().astype(int)
    
    return df

# ==========================================
# 4. STATIC PLOTS GENERATOR
# ==========================================
def generate_static_plots(df):
    print("Generating Static Plots...")
    
    def save_plot(filename):
        plt.tight_layout()
        plt.savefig(f"{config.IMAGE_DIR}/{filename}")
        plt.close()

    # --- PLOT 1: Median LOS by Procedure ---
    if not df.empty:
        average_los_by_procedure = df.groupby('procedure_name')['length_of_stay'].median().reset_index()
        average_los_by_procedure = average_los_by_procedure.sort_values('length_of_stay', ascending=False)
        average_los_by_procedure = average_los_by_procedure.reset_index(drop=True)

        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        sns.barplot(
            x='procedure_name',
            y='length_of_stay',
            data=average_los_by_procedure,
            palette='viridis',
            hue='procedure_name',
            legend=False,
            ax=ax
        )

        plt.title('Median Length of Stay by Procedure [SURGERY]', fontsize=16, fontweight='bold')
        plt.xlabel('SURGERY', fontweight='bold')
        plt.ylabel('Median Length of Stay [LOS] (Days)', fontweight='bold')

        # Label Wrapping
        labels = [item.get_text() for item in ax.get_xticklabels()]
        wrapped_labels = ['\n'.join(textwrap.wrap(label, 15)) for label in labels]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=10)

        for index, row in average_los_by_procedure.iterrows():
            ax.text(
                index,
                row['length_of_stay'],
                f"{row['length_of_stay']:.1f}",
                color='black',
                ha="center",
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        save_plot("Median_los_with_procedure.png")

    # --- PLOT 2: Readmission Types ---
    df['gap_days'] = pd.to_numeric(df['gap_days'], errors='coerce')
    df_readmits = df.dropna(subset=['gap_days']).copy()
    df_readmits = df_readmits[df_readmits['gap_days'] >= 0]
    
    if not df_readmits.empty:
        def get_category_short(days):
            if days <= 30: return "Early"
            elif days <= 90: return "Normal"
            else: return "Late"

        df_readmits['readmission_category'] = df_readmits['gap_days'].apply(get_category_short)
        order = ['Early', 'Normal', 'Late']
        type_counts = df_readmits['readmission_category'].value_counts().reindex(order).fillna(0)

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        bar_color = '#80C4E9'
        y_positions = range(len(type_counts))
        bars = ax.barh(
            y_positions,
            type_counts.values,
            color=bar_color,
            edgecolor='#222222',
            height=0.6
        )

        plt.title(f'Readmission Types (N={len(df_readmits)})', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Patients', fontweight='bold')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(type_counts.index, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        total_readmissions = len(df_readmits)
        max_count = max(type_counts.values)

        for bar in bars:
            width = bar.get_width()
            label_y = bar.get_y() + bar.get_height() / 2
            pct = (width / total_readmissions) * 100 if total_readmissions > 0 else 0
            text_str = f"{int(width)} ({pct:.1f}%)"
            ax.text(width + (max_count * 0.02), label_y, text_str, va='center', ha='left', fontsize=10, fontweight='bold', color='black')

        legend_definitions = {'Early': 'Early (≤ 30 Days)', 'Normal': 'Normal (31–90 Days)', 'Late': 'Late (> 90 Days)'}
        legend_handles = [mpatches.Patch(color=bar_color, label=legend_definitions[cat]) for cat in order]
        
        plt.legend(
            handles=legend_handles,
            title="Category Definitions",
            loc='lower right',
            fontsize=10,
            frameon=True,
            framealpha=1,
            edgecolor='black'
        )
        plt.xlim(0, max_count * 1.35)
        save_plot("readmission_categorisation.png")

    # --- PLOT 3: Readmission Gap Distribution ---
    df_gap = df[(df['gap_days'] >= 0) & (df['gap_days'] <= 90)].copy()
    if not df_gap.empty:
        mean_gap = df_gap['gap_days'].mean()
        median_gap = df_gap['gap_days'].median()
        
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        sns.histplot(
            df_gap['gap_days'],
            bins=30,
            color='#80C4E9',
            stat='percent',
            edgecolor='#222222',
            kde=True,
            line_kws={'color': '#F6FF99', 'lw': 2, 'linestyle': '--'},
            ax=ax
        )
        
        counts, bins = np.histogram(df_gap['gap_days'], bins=30)
        for i, patch in enumerate(ax.patches):
            height = patch.get_height()
            if i < len(counts) and counts[i] > 0:
                ax.annotate(f'{int(counts[i])}', xy=(patch.get_x() + patch.get_width() / 2, height), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='black')

        plt.axvline(mean_gap, color='red', linestyle='--', label=f'Mean: {mean_gap:.1f} days')
        plt.axvline(median_gap, color='green', linestyle=':', label=f'Median: {median_gap:.1f} days')
        
        plt.title(f'Readmission Gap Distribution (Within 90 Days, N={len(df_gap)})', fontsize=16, fontweight='bold')
        plt.xlabel('Days Between Discharge and Return', fontweight='bold')
        plt.ylabel('Percentage of Patients', fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(np.arange(0, 91, 10), fontsize=10)
        plt.yticks(fontsize=10)
        save_plot("readmission.png")

    # --- PLOT 4: Mortality Rate by Age Class ---
    deaths_by_class = df[df['is_death'] == 1].groupby('age_class', observed=False)['person_id'].nunique()
    total_by_class = df.groupby('age_class', observed=False)['person_id'].nunique()
    labels_ordered = ['Senior', 'Adult', 'Young Adult', 'Pediatric', 'Infant']
    y_pos = range(len(labels_ordered))
    
    mortality_rates = []
    for label in labels_ordered:
        n_deaths = deaths_by_class.get(label, 0)
        n_total = total_by_class.get(label, 0)
        pct = (n_deaths / n_total) * 100 if n_total > 0 else 0.0
        mortality_rates.append(pct)
        
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.grid(axis='x', linestyle=':', linewidth=1, alpha=0.7, zorder=0)
    bars = ax.barh(y_pos, mortality_rates, color="#80C4E9", edgecolor='#222222', zorder=3)
    
    for bar, rate, label in zip(bars, mortality_rates, labels_ordered):
        n_deaths = deaths_by_class.get(label, 0)
        n_total = total_by_class.get(label, 0)
        
        # Pct Text
        text_x = bar.get_width() / 2 
        ax.text(text_x, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}%", ha='center', va='center', color='white' if bar.get_width() > 5 else 'black', fontsize=12, fontweight='bold', zorder=4)
        
        # Raw Fraction Text
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"({n_deaths}/{n_total})", ha='left', va='center', color='#333', fontsize=10, fontstyle='italic', zorder=4)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_ordered, fontweight='bold')
    ax.set_xlabel("Mortality Rate (%)", fontweight='bold')
    ax.set_title("Mortality Rate by Age Class", fontsize=16, fontweight='bold')
    
    legend_text = "Infant: 0–1 years\nPediatric: 2–17 years\nYoung Adult: 18–39 years\nAdult: 40–59 years\nSenior: ≥ 60 years"
    ax.text(1.02, 1.0, legend_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", alpha=0.5, facecolor='white', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(f"{config.IMAGE_DIR}/mortality_age_categorisation.png")
    plt.close()

    # --- PLOT 5: Distribution of Days from Procedure to Death ---
    df_deaths = df[df['is_death'] == 1].drop_duplicates(subset=['person_id'])
    if not df_deaths.empty:
        data_series = df_deaths['days_proc_to_death'].dropna()
        total_deaths = len(data_series)
        counts, bin_edges = np.histogram(data_series, bins='fd')
        percentages = (counts / total_deaths) * 100
        
        median_days = data_series.median()
        mean_days = data_series.mean()
        max_days = data_series.max()

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.grid(axis='y', linestyle=':', linewidth=1, alpha=0.7, zorder=0)
        bars = ax.bar(bin_edges[:-1], percentages, width=np.diff(bin_edges), align='edge', color='#b3d9ff', edgecolor='black', zorder=3)

        for bar, pct in zip(bars, percentages):
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', zorder=4)

        ax.axvline(median_days, color='red', linestyle='--', linewidth=2, label=f'Median: {median_days:.1f} days', zorder=5)
        ax.axvline(mean_days, color='green', linestyle=':', linewidth=2, label=f'Mean: {mean_days:.1f} days', zorder=5)

        ax.set_title(f'Distribution of Days from Procedure to Death (Total N={total_deaths})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Days from Procedure to Death', fontweight='bold')
        ax.set_ylabel('Percentage of Deaths (%)', fontweight='bold')
        ax.set_xticks(bin_edges)
        ax.set_xticklabels([f"{x:.0f}" for x in bin_edges], rotation=45, ha='right', fontsize=9)
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
        ax.legend()

        stats_text = f"Total Deaths: {total_deaths}\nMax Survival: {max_days:.0f} days\nMedian: {median_days:.1f} days"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ccc'))

        plt.tight_layout()
        plt.savefig(f"{config.IMAGE_DIR}/days_from_procedure_to_death_hist.png")
        plt.close()

    # --- PLOT 6: LOS vs Mortality (Boxplot) ---
    df['is_death'] = pd.to_numeric(df['is_death'], errors='coerce').fillna(0).astype(int)
    df_death_plot = df[((df['is_death']==1) & (df['days_proc_to_death']>=0)) | (df['is_death']==0)].copy()
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='is_death', y='length_of_stay', data=df_death_plot, hue='is_death', legend=False, palette={0:'#80C4E9', 1:'#FFB7B2'})
    plt.title('Total LOS by Mortality Status')
    plt.xticks([0,1], ['Survivor', 'Deceased'])
    save_plot("los_vs_mortality.png")
    
    # --- PLOT 7: Post-Procedure LOS vs Mortality ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='is_death', y='post_procedure_visit_los', data=df_death_plot, hue='is_death', legend=False, palette={0:'#80C4E9', 1:'#FFB7B2'})
    plt.title('Post-Procedure LOS by Mortality Status')
    plt.xticks([0,1], ['Survivor', 'Deceased'])
    save_plot("plos_vs_mortality.png")
    
    # --- PLOT 8: LOS vs Complication ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='is_complicated', y='length_of_stay', data=df, hue='is_complicated', legend=False, palette={0:'#80C4E9', 1:'#FFB7B2'})
    plt.title('Total LOS by Complication Status')
    plt.xticks([0,1], ['No Complication', 'Complication'])
    save_plot("los_vs_complication.png")
    
    # --- PLOT 9: Post-Procedure LOS vs Complication ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='is_complicated', y='post_procedure_visit_los', data=df, hue='is_complicated', legend=False, palette={0:'#80C4E9', 1:'#FFB7B2'})
    plt.title('Post-Procedure LOS by Complication Status')
    plt.xticks([0,1], ['No Complication', 'Complication'])
    save_plot("plos_vs_complicatons.png")
    
    # --- PLOT 10: LOS vs Readmission ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='is_readmitted', y='length_of_stay', data=df, hue='is_readmitted', legend=False, palette={0:'#80C4E9', 1:'#FFB7B2'})
    plt.title('Total LOS by Readmission Status')
    plt.xticks([0,1], ['No Readmission', 'Readmitted'])
    save_plot("los_vs_readmission.png")
    
    # --- PLOT 11: Post-Procedure LOS vs Readmission ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='is_readmitted', y='post_procedure_visit_los', data=df, hue='is_readmitted', legend=False, palette={0:'#80C4E9', 1:'#FFB7B2'})
    plt.title('Post-Procedure LOS by Readmission Status')
    plt.xticks([0,1], ['No Readmission', 'Readmitted'])
    save_plot("plos_vs_readmission.png")
    
    # --- PLOT 12: Frequency Distribution of Length of Stay ---
    
    # Calculate stats first
    los_df = df  
    mean_los = los_df['length_of_stay'].mean()
    median_los = los_df['length_of_stay'].median()

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Perform the histogram plot
    sns.histplot(
        los_df['length_of_stay'],
        bins=30,
        color='#80C4E9',
        stat='percent',
        edgecolor='#222222',
        kde=True,
        line_kws={'color': '#F6FF99', 'lw': 2, 'linestyle': '--'},
        ax=ax
    )

    # Calculate actual counts per bin (for annotation)
    counts, bins = np.histogram(los_df['length_of_stay'], bins=30)

    # Annotate each bar
    for i, patch in enumerate(ax.patches):
        height = patch.get_height()
        if i < len(counts) and counts[i] > 0: 
            ax.annotate(f'{counts[i]}',
                        xy=(patch.get_x() + patch.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color='black')

    plt.axvline(mean_los, color='red', linestyle='--', label=f'Mean LOS: {mean_los:.2f} days')
    plt.axvline(median_los, color='green', linestyle=':', label=f'Median LOS: {median_los:.2f} days')

    plt.title('Frequency Distribution of Length of Stay [LOS]', fontsize=16, fontweight='bold')
    plt.xlabel('Length of Stay (Days)')
    plt.ylabel('Percentage of Patients')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Save as the filename expected by HTML
    save_plot("percentage_distribution_los.png")
    
    # --- PLOT 13: Length of Stay vs Age ---
    
    # Data Prep
    df_visits = df.copy()
    
    def get_gender_label(g):
        s = str(g).lower()
        if s in ['male', 'm', '8507', '1']: return 'Male'
        if s in ['female', 'f', '8532', '0']: return 'Female'
        return 'Other'

    df_visits['gender_label'] = df_visits['gender'].apply(get_gender_label)
    df_visits = df_visits[df_visits['gender_label'].isin(['Male', 'Female'])]
    
    PLOT_COLORS = {'Female': '#3b528b', 'Male': '#5ec962'}

    plt.figure(figsize=(14, 7))

    sns.scatterplot(
        data=df_visits,
        x='age_at_procedure', # Using 'age_at_procedure' from master query
        y='length_of_stay',
        hue='gender_label',
        palette=PLOT_COLORS,
        alpha=0.6,
        edgecolor='white',
        s=60
    )

    plt.title(f'Length of Stay vs Age (Total Inpatient Admissions: N={len(df_visits)})', fontsize=16, fontweight='bold')
    plt.xlabel('Age at Admission', fontweight='bold')
    plt.ylabel('Length of Stay (Days)', fontweight='bold')

    plt.legend(loc='upper left', title='Gender', frameon=True)

    # Counts Box
    male_visits = df_visits[df_visits['gender_label'] == 'Male'].shape[0]
    female_visits = df_visits[df_visits['gender_label'] == 'Female'].shape[0]
    text_str = f"Male Admissions: {male_visits}\nFemale Admissions: {female_visits}"

    plt.text(
        0.98, 0.96, text_str,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black')
    )

    plt.grid(True, linestyle='--', alpha=0.5)
    # Save as the filename expected by HTML
    save_plot("length of each patient.png")

    print("Static plots generated.")

# ==========================================
# 5. INTERACTIVE PLOTS GENERATOR
# ==========================================
def generate_interactive_plots(df):
    print("Generating Interactive Plots...")
    df["measurement_datetime"] = pd.to_datetime(df["measurement_datetime"])
    df["procedure_datetime"] = pd.to_datetime(df["procedure_datetime"])
    df["day_from_procedure"] = (df["measurement_datetime"].dt.normalize() - df["procedure_datetime"].dt.normalize()).dt.days
    df = df.rename(columns={"labmarker_value": "value"})
    df = df.dropna(subset=["day_from_procedure", "value", "los_days"])

    unique_procs = df['procedure_concept_id'].unique()
    unique_meas = df['measurement_concept_id'].unique()

    for proc in unique_procs:
        for meas in unique_meas:
            subset = df[(df['procedure_concept_id'] == proc) & (df['measurement_concept_id'] == meas)]
            if subset.empty: continue
            
            fig = px.line(subset, x="day_from_procedure", y="value", color="los_days", 
                          line_group="person_id", hover_data=["person_id", "los_days"],
                          title=f"Proc: {proc} | Meas: {meas}", markers=True)
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Surgery")
            fig.write_html(f"{config.INTERACTIVE_PLOT_DIR}/{int(proc)}_{int(meas)}.html", include_plotlyjs='cdn')
    
    print(f"Interactive plots saved to {config.INTERACTIVE_PLOT_DIR}")

# ==========================================
# 6. HTML REPORT GENERATION
# ==========================================
def build_report(data_dict):
    print("Compiling Final HTML Report...")
    
    env = Environment(loader=FileSystemLoader('input'))
    template = env.get_template('template.html')
    
    html_out = template.render(
        summary_rows=data_dict["summary"].to_dict(orient='records'),
        procedure_rows=data_dict["procedures"].to_dict(orient='records'),
        lab_freq_rows=data_dict["lab_freq"].to_dict(orient='records'),
        mortality_rows=data_dict["mortality"].to_dict(orient='records'),
        lab_appendix_rows=data_dict["lab_appendix"].to_dict(orient='records'),
        generation_date=pd.Timestamp.now().strftime("%Y-%m-%d")
    )
    
    with open(f"{config.OUTPUT_DIR}/final_report.html", "w", encoding="utf-8") as f:
        f.write(html_out)
    
    print(f"Report ready: {config.OUTPUT_DIR}/final_report.html")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    setup_directories()
    
    # 1. Fetch ALL Data
    data = fetch_data()
    
    # 2. Process/Clean Data
    data["master"] = process_master_data(data["master"])
    
    # 3. Generate Plots
    generate_static_plots(data["master"])
    generate_interactive_plots(data["labs_interactive"])
    
    # 4. Build HTML
    build_report(data)