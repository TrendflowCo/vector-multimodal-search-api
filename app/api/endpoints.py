from flask import Flask
from flask_caching import Cache
from flasgger import Swagger
from flask_cors import CORS
from app.config import CACHE_TYPE, CACHE_DEFAULT_TIMEOUT, SWAGGER_CONFIG
from app.api.routes import bp as api_bp

def create_app():
    """
    Create and configure an instance of the Flask application.
    """
    app = Flask(__name__)
    
    # Configure cache
    app.config['CACHE_TYPE'] = CACHE_TYPE
    app.config['CACHE_DEFAULT_TIMEOUT'] = CACHE_DEFAULT_TIMEOUT
    
    # Initialize Swagger
    Swagger(app, config=SWAGGER_CONFIG)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
