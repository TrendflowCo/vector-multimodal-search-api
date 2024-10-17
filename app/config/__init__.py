import os
from dotenv import load_dotenv

# Clear specific variables
for var in ['WEAVIATE_URL', 'WEAVIATE_API_KEY', 'WEAVIATE_CLASS_NAME_ITEMS', 'TAGS_THRESHOLD', 'MAX_K', 'SEARCH_THRESHOLD', 'GOOGLE_APPLICATION_CREDENTIALS']:
    os.environ.pop(var, None)

# Load environment variables from .env file
load_dotenv()

# Now, set the variables using os.getenv()
VERSION = os.getenv('VERSION', '1')
CACHE_DIR = os.getenv('CACHE_DIR', './cache')
CACHE_TYPE = os.getenv('CACHE_TYPE', 'SimpleCache')
CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', 86400))

SWAGGER_CONFIG = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/",
}

WEAVIATE_CLASS_NAME_ITEMS = os.getenv('WEAVIATE_CLASS_NAME_ITEMS')
WEAVIATE_CLASS_NAME_IMAGES = os.getenv('WEAVIATE_CLASS_NAME_IMAGES')
print('WEAVIATE_CLASS_NAME', WEAVIATE_CLASS_NAME_ITEMS)

TAGS_THRESHOLD = float(os.getenv('TAGS_THRESHOLD', 0.24))
MAX_K = int(os.getenv('MAX_K', 500))
SEARCH_THRESHOLD = float(os.getenv('SEARCH_THRESHOLD', 0.5))

PROPERTIES = os.getenv('PROPERTIES', '').split(',')
WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

print('PROPERTIES', PROPERTIES)
print('WEAVIATE_URL', WEAVIATE_URL)

# BigQuery configurations
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BIGQUERY_PROJECT_ID = os.getenv('BIGQUERY_PROJECT_ID', 'dokuso')
BIGQUERY_DATASET = os.getenv('BIGQUERY_DATASET', 'production')
BIGQUERY_TABLE_COMBINED = os.getenv('BIGQUERY_TABLE_COMBINED', 'combined')
BIGQUERY_TABLE_IMAGES_TAGS = os.getenv('BIGQUERY_TABLE_IMAGES_TAGS', 'images_tags')

print('GOOGLE_APPLICATION_CREDENTIALS', GOOGLE_APPLICATION_CREDENTIALS)
print('BIGQUERY_PROJECT_ID', BIGQUERY_PROJECT_ID)
