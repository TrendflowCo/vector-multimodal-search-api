import os

VERSION =os.getenv('VERSION', '1')
CACHE_DIR =os.getenv('CACHE_DIR', './cache')
CACHE_TYPE =os.getenv('CACHE_TYPE', 'SimpleCache')
CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', 86400))

WEAVIATE_CLASS_NAME =os.getenv('WEAVIATE_CLASS_NAME', 'Clipfeaturesfull')
print('WEAVIATE_CLASS_NAME', WEAVIATE_CLASS_NAME)

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

TAGS_THRESHOLD = float(os.getenv('TAGS_THRESHOLD', 0.24))
MAX_K = int(os.getenv('MAX_K', 500))

PROPERTIES = os.getenv('PROPERTIES', '').split(',')