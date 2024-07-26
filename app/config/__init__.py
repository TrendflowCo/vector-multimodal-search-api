import os

VERSION = os.environ.get('VERSION', '1')
CACHE_DIR = os.environ.get('CACHE_DIR', './cache')
CACHE_TYPE = os.environ.get('CACHE_TYPE', 'SimpleCache')
CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', 86400))

WEVIATE_CLASS_NAME = os.environ.get('WEAVIATE_CLASS_NAME', 'Clipfeaturesnew')

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

TAGS_THRESHOLD = float(os.environ.get('TAGS_THRESHOLD', 0.24))
MAX_K = int(os.environ.get('MAX_K', 500))

PROPERTIES = os.getenv('PROPERTIES', '').split(',')