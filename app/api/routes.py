from functools import lru_cache
import validators
import logging
import traceback
from flask import Blueprint, request, jsonify, make_response
from app.services.weviate_service import WeaviateService
from app.common.exceptions import InvalidParameterError, DataNotFoundError, ProcessingError
from werkzeug.exceptions import HTTPException
from app.localization import translations
from app.common.utilities import str_to_bool
from app.config import WEVIATE_CLASS_NAME  # Import the WEVIATE_CLASS_NAME
from app.data.filter_builder import FilterBuilder

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Weaviate service
weaviate_service = WeaviateService()

bp = Blueprint('api', __name__)

@lru_cache(maxsize=None)
def get_cached_similar_images(image_url, top_k):
    return weaviate_service.get_similar_items(image_url, top_k)
@lru_cache(maxsize=None)
def search_cached(properties, query, page, limit, threshold, search_type, filters):
    return weaviate_service.search_with_text(properties=properties,
                                            query_text=query, 
                                            threshold=threshold, 
                                            search_type=search_type, 
                                            filters=filters, 
                                            limit=limit, 
                                            offset=(page - 1) * limit)


@bp.route('/similar_images', methods=['GET'])
def get_similar_images():
    """
    Get similar images based on the provided image URL.

    ---
    parameters:
    - name: imageUrl
      in: query
      type: string
      required: true
      description: URL of the image to find similar images for.

    - name: top_k
      in: query
      type: integer
      required: false
      description: Number of similar images to retrieve (default 10, max 100).

    responses:
    200:
      description: Similar images retrieved successfully.
      schema:
        type: object
        properties:
          message:
            type: string
            description: Success message.
          data:
            type: object
            description: Retrieved similar images data.
    """
    try:
        image_url = request.args.get('imageUrl')
        # Validate `image_url` to ensure it is a valid URL
        if not validators.url(image_url):
            raise InvalidParameterError('Invalid URL provided')
        
        # Log the request data for better traceability
        logging.info(f"Request data - image_url: {image_url}")
        
        # Sanitize the `top_k` parameter to ensure it is within an acceptable range
        top_k = min(max(request.args.get('top_k', default=10, type=int), 1), 100)
        
        similar_images = get_cached_similar_images(image_url, top_k)
        
        return jsonify({'message': 'Similar images retrieved successfully', 'data': similar_images})
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}\n{traceback.format_exc()}")
        raise ProcessingError('An error occurred while processing the request for similar images')

@bp.route('/search', methods=['GET'])
def search():
    """
    Search for products based on various filters.

    ---
    parameters:
      - name: query
        in: query
        type: string
        required: false
        description: Search query.
      - name: imageUrl
        in: query
        type: bool
        required: false
        description: Search image.
      - name: threshold
        in: query
        type: number
        required: false
        description: Search threshold.
      - name: maxPrice
        in: query
        type: integer
        required: false
        description: Maximum price.
      - name: minPrice
        in: query
        type: integer
        required: false
        description: Minimum price.
      - name: category
        in: query
        type: string
        required: false
        description: Product category.
      - name: onSale
        in: query
        type: boolean
        required: false
        description: Filter by on sale status.
      - name: tags
        in: query
        type: string
        required: false
        description: Tags to filter by.
      - name: brands
        in: query
        type: string
        required: false
        description: Brands to filter by.
      - name: ids
        in: query
        type: string
        required: false
        description: List of product IDs to filter by.
      - name: language
        in: query
        type: string
        required: false
        description: Language for results (default en).
      - name: page
        in: query
        type: integer
        required: false
        description: Page number for pagination (default 1).
      - name: limit
        in: query
        type: integer
        required: false
        description: Number of results per page (default 100).
      - name: sortBy
        in: query
        type: string
        required: false
        description: Comma-separated list of fields to sort by.
      - name: ascending
        in: query
        type: boolean
        required: false
        description: Sort results in ascending order (default false).
      - name: search_type
        in: query
        type: string
        required: false
        description: Search type (default bm25).
      - name: threshold
        in: query
        type: number
        required: false
        description: Search threshold (default 0.5).
      - name: country
        in: query
        type: string
        required: false
        description: Country for filtering products.
      - name: currency
        in: query
        type: string
        required: false
        description: Currency for filtering products.
    responses:
      200:
        description: List of filtered products.
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object  # Define your result structure here
            page:
              type: integer
            limit:
              type: integer
            total_results:
              type: integer
            total_pages:
              type: integer
            metadata:
              type: object
              properties:
                brands:
                  type: array
                  items:
                    type: string
                min_price:
                  type: number
                max_price:
                  type: number
                tags:
                  type: array
                  items:
                    type: string
    """
    try:
        query = request.args.get('query', type=str)
        page = request.args.get('page', default=1, type=int)
        limit = request.args.get('limit', default=100, type=int)
        sort_by = request.args.get('sortBy', type=str)
        ascending = request.args.get('ascending', default='false')
        ascending = str_to_bool(ascending)  # Convert to boolean
        threshold = request.args.get('threshold', default=0.8, type=float)  # Default threshold for vector search
        max_price = request.args.get('maxPrice', type=int)
        min_price = request.args.get('minPrice', type=int)
        category = request.args.get('category', type=str)
        on_sale = request.args.get("onSale", default='false')
        on_sale = str_to_bool(on_sale)  # Convert to boolean
        tags = request.args.get('tags', type=str)
        brands = request.args.get('brands', type=str)
        list_ids = request.args.get('ids', type=str)
        search_type = request.args.get('search_type', default='clip', type=str)
        country = request.args.get('country', default='es', type=str)
        language = request.args.get('language', default='es', type=str)
        currency = request.args.get('currency', type=str)

        if not query:
            return make_response(jsonify({'error': 'Query parameter is required'}), 400)

        params = {
            'min_price': min_price,
            'max_price': max_price,
            'brands': brands,
            'tags': tags,
            'category': category,
            'on_sale': on_sale,
            'list_ids': list_ids,
            'country': country,
            'currency': currency
        }
        filters = FilterBuilder.build_filters(params)
        
        # Define properties
        properties = ['id_item',
         'shop_link',
         'brand',
         'name',
         'desc_1',
         'desc_2',
         'category',
         'price',
         'old_price',
         'country',
         'language',
         'currency',
         'discount_rate',
         'sale',
         'id_img',
         'img_url',
         'tags']
    
        
        # Execute search with Weaviate
        results = weaviate_service.search_with_text(
            properties=properties,
            query_text=query, 
            threshold=threshold, 
            search_type=search_type, 
            filters=filters, 
            limit=limit, 
            offset=(page - 1) * limit,
            sort_by=sort_by,
            ascending=ascending
        )

        # Process results for response
        results_data = results.get('data', {}).get('Get', {}).get(WEVIATE_CLASS_NAME, [])
        if not results_data:
            return make_response(jsonify({'error': 'No results found'}), 404) 
        
        total_results = len(results_data)  # This might need adjustment based on actual Weaviate response
        total_pages = (total_results + limit - 1) // limit

        # Translate tags if necessary
        all_tags = []
        for item in results_data:
            if 'tags' in item:  # Ensure that the 'tags' key exists in the item
                for tag in item['tags']:
                    # Translate each tag and append to all_tags list
                    all_tags.append(translations.tags[language].get(tag, tag))

        return jsonify({
            'results': results_data,
            'page': page,
            'limit': limit,
            'total_results': total_results,
            'total_pages': total_pages,
            'metadata': {
                'brands': list(set([item['brand'] for item in results_data])),
                'min_price': min(item['price'] for item in results_data) if results_data else None,
                'max_price': max(item['price'] for item in results_data) if results_data else None,
                'tags': all_tags
                }
            })
    except Exception as e:
        logging.error(f"Search failed: {str(e)}\n{traceback.format_exc()}")
        raise ProcessingError('An error occurred while processing the search request')

@bp.route('/product', methods=['GET'])
def product():
    """
    Get product details by ID.

    ---
    parameters:
      - name: id
        in: query
        type: string
        required: true
        description: Product ID.
      - name: language
        in: query
        type: string
        required: false
        description: Language for results (default en).
      - name: country
        in: query
        type: string
        required: false
        description: Country for filtering product details.
      - name: currency
        in: query
        type: string
        required: false
        description: Currency for filtering product details.
    responses:
      200:
        description: Product details.
        schema:
          type: object  # Define your result structure here
          properties:
            result:
              type: object
    """
    try:
        product_id = request.args.get('id')
        language = request.args.get('language', default='en')
        sort_by = request.args.get('sortBy', type=str)
        ascending = request.args.get('ascending', default='false')
        ascending = str_to_bool(ascending)  # Convert to boolean
        country = request.args.get('country', type=str)
        currency = request.args.get('currency', type=str)

        if not product_id:
            return make_response(jsonify({'error': 'Product ID is required'}), 400)

        # Define properties
        properties = ["id_item", "name", "price", "brand", "tags", "img_url"]
        
        # Prepare filters using FilterBuilder
        params = {
            'country': country,
            'currency': currency
        }
        filters = FilterBuilder.build_filters(params)

        # Retrieve product details using Weaviate
        result = weaviate_service.get_product_details(
            properties, product_id, language, sort_by, ascending, filters
        )

        # Check if the product was found
        if not result.get('data', {}).get('Get', {}).get(WEVIATE_CLASS_NAME):
            raise DataNotFoundError('Product not found')

        # Extract product data
        product_data = result['data']['Get'][WEVIATE_CLASS_NAME][0]

        # Translate fields if necessary
        if 'tags' in product_data:
            product_data['tags'] = [translations.tags[language].get(tag, tag) for tag in product_data['tags']]

        return jsonify({
            'result': product_data
        })
    except Exception as e:
        logging.error(f"Product details retrieval failed: {str(e)}\n{traceback.format_exc()}")
        raise ProcessingError('An error occurred while retrieving product details')

@bp.route('/most_similar_items', methods=['GET'])
def most_similar_items():
    """
    Get a list of most similar items to a given product.

    ---
    parameters:
      - name: id
        in: query
        type: string
        required: true
        description: Product ID.
      - name: top_k
        in: query
        type: integer
        required: false
        description: Number of similar items to retrieve (default 20).
      - name: country
        in: query
        type: string
        required: false
        description: Country for filtering similar items.
      - name: currency
        in: query
        type: string
        required: false
        description: Currency for filtering similar items.
    responses:
      200:
        description: List of most similar items.
        schema:
          type: object  # Define your result structure here
          properties:
            results:
              type: array
              items:
                type: object
    """
    try:
        product_id = request.args.get('id')
        top_k = request.args.get('top_k', default=20, type=int)
        sort_by = request.args.get('sortBy', type=str)
        ascending = request.args.get('ascending', default='false')
        ascending = str_to_bool(ascending)  # Convert to boolean
        country = request.args.get('country', type=str)
        currency = request.args.get('currency', type=str)

        if not product_id:
            return make_response(jsonify({'error': 'Product ID is required'}), 400)

        # Prepare filters using FilterBuilder
        params = {
            'country': country,
            'currency': currency
        }
        filters = FilterBuilder.build_filters(params)

        # Retrieve similar items using Weaviate
        results = weaviate_service.get_similar_items(
            product_id, top_k, sort_by, ascending, filters
        )

        if not results:
            return make_response(jsonify({'error': 'No similar items found or ID not found'}), 404)

        # Extract results data
        similar_items = results.get('data', {}).get('Get', {}).get(WEVIATE_CLASS_NAME, [])

        return jsonify({
            'results': similar_items
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/image_query_similarity', methods=['GET'])
def image_query_similarity():
    """
    Get similarity score between an image query and an image URL.

    ---
    parameters:
      - name: query
        in: query
        type: string
        required: true
        description: Image query.
      - name: img_url
        in: query
        type: string
        required: true
        description: Image URL.
    responses:
      200:
        description: Similarity score.
        schema:
          type: object  # Define your result structure here
          properties:
            similarity_score:
              type: number
    """
    query = request.args.get('query', type=str)
    img_url = request.args.get('img_url', type=str)
    

    if all([i is None for i in [query, img_url]]):
        return make_response(jsonify({'error': 'Missing parameter'}), 400)

    score, e = get_image_query_similarity_score(query, img_url)
    
    if not score:
        logging.error(f"Failed to compute image query similarity: {str(e)}\n{traceback.format_exc()}")
        return make_response(jsonify({'error': 'Failed to compute image query similarity'}), 400)
       
    return jsonify({
        'similarity_score': score
    })

@bp.route('/brands_list', methods=['GET'])
def brands():
    """
    Get the list of all brands in catalogue.

    ---
    parameters:
    responses:
      200:
        description: List of brands in catalogue.
        schema:
          type: object  # Define your result structure here
          properties:
            brand_list:
              type: array
    """
    try:
        # Retrieve all brands using Weaviate
        brands_list = weaviate_service.get_all_brands()

        if not brands_list:
            return make_response(jsonify({'error': 'No brands found'}), 404)

        return jsonify({
            'brand_list': brands_list
        })
    except Exception as e:
        logging.error(f"Failed to list brands: {str(e)}\n{traceback.format_exc()}")
        return make_response(jsonify({'error': 'Failed to list brands'}), 400)
