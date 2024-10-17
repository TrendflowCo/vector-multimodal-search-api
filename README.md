# TrendFlow API 🔍👗

Private API powering TrendFlow, the cutting-edge fashion multimodal search platform.

## 🌟 Key Features

- Text and image-based fashion search
- Multi-language support
- Advanced filtering (brand, price, category, sale status)
- Real-time data ingestion from BigQuery
- High-performance vector search with Weaviate
- Secure API key authentication
- Rate limiting for API protection

## 🛠️ Tech Stack

- Python 3.11
- Flask
- CLIP (OpenAI)
- Weaviate
- Google BigQuery
- Docker
- Flask-HTTPAuth
- Flask-Limiter

## 🚀 Quick Start (For Development)

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Generate API key: `python generate_api_key.py`
4. Set up environment variables: Copy `.env-example` to `.env` and fill in values, including the generated API key
5. Run the app: `python main.py`

## 🐳 Deployment

```bash
docker build -t trendflow-api .
docker run -p 8080:8080 trendflow-api
```

## 🧪 Testing

Run tests: `python -m unittest discover tests`

## 📚 API Documentation

Internal Swagger UI available at `/swagger/` (requires authentication)

## 🌐 Internationalization

Manage translations in `app/localization/translations.py`

## 🔒 Security

- API key required for all endpoints
  - Include in requests as: `Authorization: Bearer YOUR_API_KEY`
- Rate limiting enforced:
  - Search: 10 requests per minute
  - Product details: 20 requests per minute
  - Similar items: 10 requests per minute
  - Brands list: 5 requests per minute
  - Data ingestion: 2 requests per hour
- Data encryption in transit (HTTPS recommended in production)

## 📈 Monitoring

Prometheus metrics available at `/metrics` endpoint (requires authentication)

## 🤝 Support

For issues or feature requests, please contact the TrendFlow development team.

## ⚠️ Disclaimer

This is a private API. Unauthorized access or use is strictly prohibited.
