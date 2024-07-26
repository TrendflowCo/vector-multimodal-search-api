from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from app.api.endpoints import create_app

app = create_app()

if __name__ == "__main__":
    # Run the Flask application
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
