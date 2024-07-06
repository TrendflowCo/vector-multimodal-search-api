import os

VERSION = '1'
CACHE_DIR = './cache'
CACHE_TYPE = 'SimpleCache'
CACHE_DEFAULT_TIMEOUT = 3600 * 24  # 24 hours

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

TAGS_THRESHOLD = 0.24
MAX_K = 500