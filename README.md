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

API key required for all endpoints. Include in requests as:
```
Authorization: Bearer YOUR_API_KEY
```

## 📡 Endpoints

1. **Search** - `/search` (GET)
   - Search for products based on various filters
   - Parameters: query, imageUrl, threshold, maxPrice, minPrice, category, onSale, tags, brands, ids, language, page, limit, sortBy, ascending, search_type, country, currency

2. **Product Details** - `/product` (GET)
   - Get product details by ID
   - Parameters: id, language, country, currency

3. **Similar Items** - `/most_similar_items` (GET)
   - Get a list of most similar items to a given product
   - Parameters: id, top_k, country, language, currency

4. **Brands List** - `/brands_list` (GET)
   - Get the list of all brands in catalogue

5. **Image Query Similarity** - `/image_query_similarity` (GET)
   - Get similarity score between an image query and an image URL
   - Parameters: query, img_url

6. **Data Ingestion** - `/ingest` (POST)
   - Ingest data into Weaviate from BigQuery

For detailed usage of each endpoint, refer to the Swagger documentation.

<!-- ## 📈 Monitoring

Prometheus metrics available at `/metrics` endpoint (requires authentication) -->

## 🤝 Support

For issues or feature requests, please contact the TrendFlow development team.

## ⚠️ Disclaimer

This is a private API. Unauthorized access or use is strictly prohibited.
