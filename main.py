# import os

# Manually load the .env file
# with open('.env', 'r') as file:
#     for line in file:
#         if line.startswith('#') or line.strip() == '':
#             continue
#         try:    
#             key, value = line.strip().split('=', 1)
#             os.environ[key] = value
#         except ValueError:
#             print(f"Invalid line in .env file: {line}")

from dotenv import load_dotenv

load_dotenv('.env', override=True)

from app.api.endpoints import create_app
import os

app = create_app()

if __name__ == "__main__":
    # Run the Flask application
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)