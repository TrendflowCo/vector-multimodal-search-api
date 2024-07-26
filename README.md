# Fashion CLIP Search

A powerful and efficient fashion search engine powered by CLIP (Contrastive Language-Image Pre-training) technology.

## 🚀 Features

- Text-based fashion item search
- Image similarity search
- Multi-language support
- Brand filtering
- Price range filtering
- Category-based search
- Sale item filtering

## 🛠️ Technologies Used

- Python 3.11
- Flask
- CLIP (OpenAI)
- Weaviate
- Docker
- Google Cloud Platform

## 🏗️ Project Structure

The project is organized into several key components:

- `app/`: Main application directory
  - `api/`: API endpoints and routes
  - `common/`: Utility functions
  - `config/`: Configuration settings
  - `data/`: Data-related modules
  - `localization/`: Internationalization support
  - `models/`: CLIP models and related functions
  - `services/`: Business logic and services

- `notebooks/`: Jupyter notebooks for testing and development
- `tests/`: Unit tests

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env-example` to `.env`
   - Fill in your specific values in the `.env` file
4. Run the application:
   ```
   python main.py
   ```

## 🌍 Environment Variables

The project uses environment variables for configuration. A `.env-example` file is provided with all the necessary variables. Make sure to copy this file to `.env` and update the values before running the application.

## 🐳 Docker Support

The project includes Docker support for easy deployment. Build and run the Docker image using:

```
docker build -t fashion-clip-search .
docker run -p 8080:8080 fashion-clip-search
```

## 🚀 Deployment

The project is set up for deployment on Google Cloud Platform using Cloud Build and Cloud Run. Refer to the `cloudbuild.yaml` file for deployment configuration.

## 🧪 Testing

Run the unit tests using:

```
python -m unittest discover tests
```

## 📚 API Documentation

API documentation is available using Swagger. Access the Swagger UI at `/swagger/` when running the application.

## 🌐 Internationalization

The project supports multiple languages. Translations are managed in the `app/localization/translations.py` file.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
