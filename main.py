from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from app.api.endpoints import create_app

app = create_app()

if __name__ == "__main__":
    # Run the Flask application
    app.run()
