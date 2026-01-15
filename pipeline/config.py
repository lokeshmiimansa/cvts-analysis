# 1. DATABASE CREDENTIALS
DB_CONFIG = {
    "dbname": "cdm",    
    "user": "usr1",
    "password": "pass",
    "host": "localhost",      
    "port": "5432",
    "schema": "schema_name"
}

# 2. SELECTION CONFIGURATION
# These IDs will filter the data for the interactive plots
SELECTED_PROCEDURES = [
    4012932, 4019929, 4095407, 4142628, 4203153, 
    4284104, 4294387, 4302815, 4304688
]

SELECTED_MEASUREMENTS = [
    3034426, 42868738, 3049746, 3034022, 40765586, 
    3041944, 3027193, 3005872, 3050583, 36305724, 
    36303488, 3034734, 40762632, 1761868
]

# 3. PATHS
TEMPLATE_PATH = 'input/template.html'
OUTPUT_DIR = 'output'
IMAGE_DIR = 'output/images'
INTERACTIVE_PLOT_DIR = 'output/plot'
DATA_DIR = 'output/data_cache'