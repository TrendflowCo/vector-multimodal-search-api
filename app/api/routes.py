import logging
import traceback

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from flask import Blueprint, request, jsonify, make_response
from app.services.image_service import ImageService
from app.services.text_service import TextService
from app.services.similarity_service import SimilarityService
from app.localization import translations
from app.data.data_access import DataAccess
from app.data.data_processing import ImageDataProcessor, ImageIndexBuilder
from app.common.utilities import paginate_results, concatenate_embeddings, create_item_data
from app.common.exceptions import DataNotFoundError, InvalidParameterError
import torch
import pandas as pd
import numpy as np
import faiss
from werkzeug.exceptions import HTTPException

bp = Blueprint('api', __name__)

data_access = DataAccess()
image_data_processor = ImageDataProcessor(data_access)
image_index_builder = ImageIndexBuilder(image_data_processor.e_img_clip_dict, image_data_processor.images_df, pd.DataFrame([], index=image_data_processor.e_img_clip_dict.keys()))

similarity_service = SimilarityService(
    images_df=image_data_processor.images_df,
    e_img_clip_dict=image_data_processor.e_img_clip_dict,
    e_img_clip_cat=concatenate_embeddings(list(image_data_processor.e_img_clip_dict.values())),
    e_img_clip_df=pd.DataFrame([], index=image_data_processor.e_img_clip_dict.keys()),
    image_data_dict={idx: {**create_item_data(row), 'img_url': idx} for idx, row in image_data_processor.images_df.iterrows()},
    image_index_builder=image_index_builder
)

@bp.route('/similar_images/<image_url>', methods=['GET'])
def get_similar_images(image_url):
    try:
        top_k = request.args.get('top_k', default=10, type=int)
        similar_images = similarity_service.retrieve_similar_images(image_url, top_k)
        return jsonify({'message': 'Similar images retrieved successfully', 'data': similar_images})
    except Exception as e:
        logging.error(f"Failed to retrieve similar images: {str(e)}\n{traceback.format_exc()}")
        return make_response(jsonify({'error': 'Failed to retrieve similar images'}), 400)

@bp.route('/text_image_similarity', methods=['POST'])
def get_text_image_similarity():
    try:
        data = request.json
        text = data.get('text')
        image_url = data.get('image_url')
        if not text or not image_url:
            raise InvalidParameterError("Text and image URL must be provided")
        similarity_score = similarity_service.compute_similarity_between_text_and_image(text, image_url)
        return jsonify({'message': 'Similarity score calculated successfully', 'similarity_score': similarity_score})
    except InvalidParameterError as e:
        logging.error(f"Invalid parameters: {str(e)}")
        return make_response(jsonify({'error': str(e)}), 400)
    except Exception as e:
        logging.error(f"Error computing similarity: {str(e)}\n{traceback.format_exc()}")
        return make_response(jsonify({'error': 'Error computing similarity'}), 400)

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
        query = request.args.get('query')
        image_url = request.args.get('imageUrl')
        threshold = request.args.get('threshold', default=0.21, type=float)
        max_price = request.args.get('maxPrice', type=int)
        min_price = request.args.get('minPrice', type=int)
        category = request.args.get('category')
        on_sale = request.args.get("onSale", default=False, type=bool)
        tags = request.args.get('tags')
        brands = request.args.get('brands')
        list_ids = request.args.get('ids')
        language = request.args.get('language', default='en')
        page = request.args.get('page', default=1, type=int)
        limit = request.args.get('limit', default=100, type=int)
        sort_by = request.args.get('sortBy')
        ascending = request.args.get('ascending', default=False, type=bool)
        

        if all([i is None for i in [query, category, image_url, on_sale, brands, list_ids]]):
            return make_response(jsonify({'error': 'Missing parameter'}), 400)

        filters = {
            'query': query,
            'image_url': image_url,
            'threshold': threshold,
            'category': category,
            'min_price': min_price,
            'max_price': max_price,
            'tags': tags,
            'brands': brands,
            'on_sale': on_sale,
            'ids': list_ids,
            'language': language
        }

        results = similarity_service.retrieve_filtered_images(filters)

        available_brands, min_overall_price, max_overall_price = '', 0, 0

        if (results is None) or (len(results)==0):
            return make_response(jsonify({'error': 'Results no found'}), 204)
            
        available_brands = results['brand'].unique().tolist()
        min_overall_price = results['price'].min()
        max_overall_price = results['price'].max()
        all_img_urls = results['img_url'].head(30).unique().tolist()
        good_keys = image_data_processor.images_tags_df.index.intersection(all_img_urls)
        all_tags = image_data_processor.images_tags_df.loc[good_keys]['value'].explode().drop_duplicates().map(translations.tags[language]).dropna().unique().tolist()
        # all_tags = [translations.tags[language][tag] for tag in all_tags]
    

        if sort_by:
            list_sort_by = sort_by.replace("'", "").split(',')
            results = results.sort_values(by=list_sort_by, ascending=ascending)
        
        results = results.to_dict(orient='records')

        total_results, total_pages, paginated_results = paginate_results(results, page, limit)
        
        return jsonify({
            'results': paginated_results,
            'page': page,
            'limit': limit,
            'total_results': total_results,
            'total_pages': total_pages,
            'metadata': {'brands': available_brands, 
                        'min_price': min_overall_price,
                        'max_price': max_overall_price,
                        'tags': all_tags}
            })
        # return jsonify({
        #     'message': 'Search results',
        #     'data': paginated_results,
        #     'metadata': {
        #         'total_results': total_results,
        #         'total_pages': total_pages,
        #         'page': page,
        #         'limit': limit
        #     }
        # })
    except Exception as e:
        logging.error(f"Search failed: {str(e)}\n{traceback.format_exc()}")
        return make_response(jsonify({'error': 'Search failed'}), 400)

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
    responses:
      200:
        description: Product details.
        schema:
          type: object  # Define your result structure here
          properties:
            result:
              type: object
    """
    
    id = request.args.get('id')
    language = request.args.get('language', default='en')

    if all([i is None for i in [id]]):
        return make_response(jsonify({'error': 'Missing parameter'}), 400)


    result = retrieve_product_data(image_data_processor.all_images_df, id, language)
    
    return jsonify({
        'result': result
    })
       
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
      - name: threshold
        in: query
        type: integer
        required: false
        description: Number of similar items to retrieve (default 20).
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
    id = request.args.get('id', type=str)
    top_k = request.args.get('threshold', default=20, type=int)

    if all([i is None for i in [id, top_k]]):
        return make_response(jsonify({'error': 'Missing parameter'}), 400)

    results = similarity_service.retrieve_most_similar_items(id, top_k)
    
    if not results:
        return make_response(jsonify({'error': 'Id not found'}), 400)
        
    return jsonify({
        'results': results
    })

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
        return jsonify({
            'brand_list': image_data_processor.all_images_df.brand.unique().tolist()
        })
    except Exception as e:
        logging.error(f"Failed to list brands: {str(e)}\n{traceback.format_exc()}")
        return make_response(jsonify({'error': 'Failed to list brands'}), 400)
