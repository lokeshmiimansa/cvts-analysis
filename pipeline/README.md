# CVTS Outcomes Analysis Pipeline

## 🏥 Overview
This project is an automated data engineering and visualization pipeline designed for **Cardiovascular and Thoracic Surgery (CVTS)** observational data. It connects to an OMOP CDM PostgreSQL database, extracts clinical cohorts, performs statistical analysis, and generates a comprehensive HTML report containing both static publication-ready charts and interactive lab marker trends.

## ✨ Features
* **Dynamic Cohort Extraction:** extracting patient demographics, procedures, and outcomes based on OMOP Concept IDs.
* **Automated Data Cleaning:** Handles date logic, invalid death records, and age categorization.
* **Static Visualization:** Generates Matplotlib/Seaborn charts for Mortality, Length of Stay (LOS), and Readmission patterns.
* **Interactive Analysis:** Generates Plotly interactive HTML plots for specific lab markers (e.g., Hemoglobin, Creatinine) over time.
* **HTML Reporting:** Compiles all statistics and visuals into a single `final_report.html` using Jinja2 templating.
* **Client-Side Security:** Includes a basic password overlay for the generated report.

## 📂 Project Structure

```text
cvts_pipeline/
│
├── config.py               # Configuration (DB credentials, Concept IDs selections)
├── queries.py              # SQL query templates (Jinja/f-string based)
├── pipeline.py             # Main execution script (ETL + Viz + Reporting)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── input/
│   └── template.html       # HTML Jinja2 template for the report
│
└── output/                 # Generated artifacts (created automatically)
    ├── final_report.html   # The final dashboard
    ├── images/             # Static PNG charts
    ├── plot/               # Interactive HTML plots
    └── data_cache/         # CSV backups of fetched data

```

## 🚀 Installation

### 1. Prerequisites

* Python 3.8+
* PostgreSQL Database (with CVTS OMOP CDM schema)

### 2. Environment Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

## ⚙️ Configuration

1. Open `config.py`.
2. Update the **DB_CONFIG** dictionary with your PostgreSQL credentials:
```python
DB_CONFIG = {
    "dbname": "your_db",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5432",
    "schema": "cvts_cdm"
}

```


3. (Optional) Update `SELECTED_PROCEDURES` or `SELECTED_MEASUREMENTS` if you wish to analyze different cohorts.

## 🏃‍♂️ Usage

### Run the Pipeline

Execute the main script. This will fetch data, generate all plots, and compile the report.

```bash
python pipeline.py

```

### View the Report

Since the report uses local interactive plots in iframes, it is best viewed via a local web server rather than double-clicking the file.

```bash
cd output
python -m http.server 8000

```

Open your browser to: `http://localhost:8000/final_report.html`

**(Password is set in the HTML template script, default: `sgpgi2025`)**

---

## 🛠️ Customization

* **To change the Report Layout:** Edit `input/template.html`.
* **To change Plot Styles:** Edit the `generate_static_plots` function in `pipeline.py`.
* **To change Logic:** Edit SQL queries in `queries.py`.