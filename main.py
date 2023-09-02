import io
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import clip
from flask import Flask, jsonify, request, make_response
from flask_caching import Cache
from flask_cors import CORS
from joblib import Memory
import functools
import faiss
from utils import *
# from gcp_utils import *
from fetch_data import *
from templates import templates, templates_with_adjectives, garment_types
from flasgger import Swagger
import translations

# Create a cache for storing text embeddings
cache_dir = './cache'
memory = Memory(cache_dir, verbose=0)


swagger_config = {
    "headers": [
    ],
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

app = Flask(__name__)
cache = Cache(app)
swagger = Swagger(app, config=swagger_config)
CORS(app)

# Set up cache configuration
app.config['CACHE_TYPE'] = 'SimpleCache'
# app.config['CACHE_DIR'] = './app_cache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600*24 # 5 minutes
app.config['SWAGGER'] = {
    'title': 'Fashion CLIP Search API',
    'uiversion': 3
}


# Load the CLIP model
model, preprocess = clip.load("ViT-L/14@336px")
model.cpu().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size
        
blacklist = ['https://www.stradivarius.com/es/coletero-grande-saten-l00150411']


query = """SELECT  
    distinct
    a.id, 
    a.shop_link, 
    a.brand, 
    a.name,
    a.desc_1,
    a.desc_2,
    a.category, 
    a.price,
    a.old_price, 
    a.currency,
    a.discount_rate, 
    a.sale, 
    b.img_url
FROM `dokuso.listing_products.items` a
LEFT JOIN `dokuso.listing_products.images` b
    ON a.id = b.id_item
INNER JOIN (
    SELECT id, MAX(updated_at) as latest_update
    FROM `dokuso.listing_products.items`
    GROUP BY id
) latest_items
    ON a.id = latest_items.id AND a.updated_at = latest_items.latest_update
WHERE b.img_url IS NOT NULL"""

images_df = query_datasets_to_df(query).drop_duplicates(['img_url']).set_index('img_url')


query = f"""
SELECT
b.img_url,
a.features
from `dokuso.embeddings.images` a
left join `dokuso.listing_products.images` b
on a.id_img = b.id
"""

e_img_clip = query_datasets_to_df(query).set_index('img_url')['features'].T.to_dict()

query = f"""
SELECT
b.img_url,
a.features
from `dokuso.embeddings.images_fclip` a
left join `dokuso.listing_products.images` b
on a.id_img = b.id
"""

e_img_fclip = query_datasets_to_df(query).set_index('img_url')['features'].T.to_dict()

query = f"""
SELECT
a.img_id, 
a.category, 
a.tags,
b.img_url
from `dokuso.tagging.images`a
left join `dokuso.listing_products.images` b
on a.img_id = b.id
"""

images_tags_df = query_datasets_to_df(query).set_index('img_url')

blacklist_img_url = images_df[images_df['shop_link'].isin(blacklist)].index.tolist()


# Filtering out blacklisted images
e_img_clip = {k: torch.tensor(v).reshape(1, -1).float() for k,v in e_img_clip.items() if v is not None}

list_urls = list(e_img_clip.keys())

e_img_fclip = {k: torch.tensor(v) for k,v in e_img_fclip.items() if (v is not None) and (k in list_urls)}

images_df['price'] = images_df['price'].fillna(0)
images_df['old_price'] = images_df['old_price'].fillna(images_df['price'])
images_df['discount_rate'] = images_df['discount_rate'].fillna(0)
all_images_df = images_df.copy()

images_df = images_df[images_df.index.isin(list_urls)]

e_img_clip_cat = torch.cat(list(e_img_clip.values()))
e_img_clip_df = pd.DataFrame([], index=e_img_clip.keys())

# Prepare the embeddings for FAISS
embeddings = np.array([embedding.numpy().flatten() for embedding in e_img_clip.values()], dtype=np.float32)

# Build an index with the embeddings
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
index.add(embeddings)


@memory.cache
def compute_text_embeddings(text):
    text_tokens = clip.tokenize([text]).cpu()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).cpu().float()
    return text_features

@functools.lru_cache(maxsize=None)
def get_similarity(image_features, text_features):
    image_features_normalized = F.normalize(image_features, dim=1)
    text_features_normalized = F.normalize(text_features, dim=1)
    similarity = text_features_normalized @ image_features_normalized.T
    return similarity

def clean_prompt_text(prompt):
    if prompt.startswith('a '):
        prompt = prompt[2:]
    if prompt.startswith('an '):
        prompt = prompt[3:]
    if prompt.startswith('the '):
        prompt = prompt[4:]
    prompt = prompt.lower().replace('style', '')
    prompt = prompt.replace('fashion', '')
    prompt = prompt.replace('inspiration', '')
    prompt = prompt.lower()
    return prompt


def generate_texts(prompt, templates):
    if any([x in garment_types for x in prompt.split(' ')]):
        return [template.format(prompt) for template in templates]
    else:
        return [template.format(prompt) for template in templates_with_adjectives]


def get_image_query_similarity_score(query, img_url):
    try:
        query = clean_prompt_text(query)
        texts = generate_texts(query, templates)
        e_text_cat = torch.cat([compute_text_embeddings(t) for t in texts]).to('cpu')
        similarity_score = get_similarity(e_img_clip[img_url], e_text_cat).max().item()
        return similarity_score, None
    except Exception as e:
        return None, str(e)

def compute_similarities(e_img_clip_cat, query, max_k, threshold, blacklist_img_url):
    texts = generate_texts(query, templates)
    e_text_cat = torch.cat([compute_text_embeddings(t) for t in texts]).to('cpu')
    similarity = get_similarity(e_img_clip_cat, e_text_cat)
    if max_k is not None:
        top_idx = np.array(list(set(similarity.topk(max_k).indices.ravel().tolist())))
    else:
        top_idx = np.arange(similarity.shape[1])  # All indices
    print('Threshold', threshold)
    top_idx = top_idx[(similarity[:, top_idx].max(0).values > threshold).numpy()]

    similar_items = []

    for idx in top_idx:
        idx = idx.item()
        idx_ = e_img_clip_df.index[idx]
        item_data = image_data_dict[idx_]
        similarity_max = similarity[:, idx].max().item()
        item_data['similarity'] = similarity_max
        similar_items.append(item_data)

    return similar_items

def create_item_data(row):
    return {
        'id': row['id'],
        'brand': row['brand'],
        'name': row.get('name'),
        'category': row.get('category'),
        # 'desc_1': row.get('desc_1'),
        # 'desc_2': row.get('desc_2'),
        'img_url': row.get('img_url'),
        'price': row.get('price'),
        'old_price': row.get('old_price'),
        'currency': row.get('currency'),
        'discount_rate': row.get('discount_rate'),
        'sale': bool(row.get('sale')),
        'shop_link': row.get('shop_link')
    }


# Create a dictionary to store image data for faster lookups
image_data_dict = {}
for idx, row in images_df.iterrows():
    image_data_dict[idx] = create_item_data(row)
    image_data_dict[idx]['img_url'] = idx


def retrieve_filtered_images(images_df, filters, language):
    filtered_items = images_df.copy()

    if filters.get('query'):
        query = filters['query']
        threshold = filters.get('threshold', 0.21)
        if language != 'en':
            print(f"Translation from {language} to en: \n")
            print(f"Original: {query}\n")
            query = translate(query, language)
            print(f"English: {query}")
        similar_items = compute_similarities(e_img_clip_cat, query, max_k=None, threshold=threshold, blacklist_img_url=None)
        if len(similar_items) == 0:
            return None
        similar_items_df = pd.DataFrame(similar_items).set_index('img_url')
        similar_items_df = similar_items_df.sort_values(by='similarity', ascending=False)
        filtered_items = filtered_items.loc[similar_items_df.index]
        filtered_items['similarity'] = similar_items_df['similarity']

    if filters.get('category'):
        filtered_items = filtered_items[filtered_items['category'] == filters['category']]
    if filters.get('min_price'):
        filtered_items = filtered_items[filtered_items['price'] >= filters['min_price']]
    if filters.get('max_price'):
        filtered_items = filtered_items[filtered_items['price'] <= filters['max_price']]
    if filters.get('brands'):
        list_brands = [x.lower() for x in filters['brands'].replace("'", "").split(',')]
        filtered_items = filtered_items[filtered_items['brand'].str.lower().isin(list_brands)]
    if filters.get('on_sale'):
        filtered_items = filtered_items[filtered_items['sale'] == filters['on_sale']]
    if filters.get('ids'):
        list_ids = filters['ids'].replace("'", "").split(',')
        filtered_items = filtered_items[filtered_items['id'].isin(list_ids)]

    filtered_items.reset_index(inplace=True)

    return filtered_items.drop_duplicates(['img_url', 'shop_link'])

def retrieve_most_similar_items(input_id, k):
    
    input_index = images_df[images_df['id'] == input_id].index

    if len(input_index)==0:
        return None
    
    input_img_url = input_index[0]
        
    input_shop_link =  images_df.loc[input_img_url]['shop_link']
    
    input_embedding = e_img_clip[input_img_url].numpy().flatten()
    
    _, indices = index.search(np.array([input_embedding], dtype=np.float32), k + 1)

    similar_images = []
    shop_link_similarities = {}
    for i, idx in enumerate(indices[0][1:]):
        img_url = list_urls[idx]
        row = images_df.loc[img_url]
        row['img_url'] = img_url
        shop_link = row['shop_link']
        if shop_link == input_shop_link:
            continue
        similarity_score = 1 - i / k
        
        if shop_link in shop_link_similarities:
            if similarity_score > shop_link_similarities[shop_link]:
                shop_link_similarities[shop_link] = similarity_score
                item_data = create_item_data(row)
                item_data['similarity'] = similarity_score
                similar_images.append(item_data)
        else:
            item_data = create_item_data(row)
            item_data['similarity'] = similarity_score
            similar_images.append(item_data)
    
    return similar_images

def paginate_results(results, page, limit):
    total_results = len(results)
    total_pages = (total_results - 1) // limit + 1
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_results = results[start_index:end_index]

    return total_results, total_pages, paginated_results

def get_image_top_tags(img_url, THRESHOLD=0.25):
    
    top_tags_df = tags_df.copy()

    col_name = f'similarity_{col_idx}'
    top_tags_df[col_name] = pd.Series(tags_similarities_dict[img_url])
    top_tags_df = top_tags_df[top_tags_df[col_name]>0.2]
    p = top_tags_df.groupby('category')[col_name].rank(pct=True)
    p.name = f'percentile_{col_idx}'
    top_tags_df = top_tags_df.join(p)
    top_tags_df[f'score_{col_idx}'] = top_tags_df[col_name] * top_tags_df[p.name]
    
    # Filter and sort based on score
    filtered_df = top_tags_df[top_tags_df[f'score_{col_idx}'] > THRESHOLD].sort_values(by=f'score_{col_idx}', ascending=False)
    
    # Create a dictionary of top tags for each category and each similarity column
    tags_by_category = filtered_df.groupby('category')['value'].agg(set).map(lambda x: list(x))
    tags_by_category = tags_by_category.to_dict()

    return tags_by_category


def create_specific_item_data(item_data, language='en'):

    good_keys = images_tags_df.index.intersection(item_data['img_url'].tolist())
    all_tags = images_tags_df.loc[good_keys]['tags'].explode().drop_duplicates().map(translations.tags[language]).dropna().unique().tolist()
    # all_tags = [translations.tags[language][tag] for tag in all_tags]
    return {
        'id': item_data.iloc[0]['id'],
        'brand': item_data.iloc[0]['brand'],
        'name': item_data.iloc[0]['name'],
        'category': item_data.iloc[0]['category'],
        'desc_1': item_data.iloc[0]['desc_1'],
        'desc_2': item_data.iloc[0]['desc_2'],
        'img_url': item_data['img_url'].tolist(),
        'price': item_data.iloc[0]['price'],
        'old_price': item_data.iloc[0]['old_price'],
        'currency': item_data.iloc[0]['currency'],
        'discount_rate': item_data.iloc[0]['discount_rate'],
        'sale': bool(item_data.iloc[0]['sale']),
        'shop_link': item_data.iloc[0]['shop_link'],
        'tags': all_tags
    }

def retrieve_product_data(images_df, id, language='en'):
    item_data = images_df[images_df['id'] == id].reset_index()
    if item_data is not None:
        return create_specific_item_data(item_data, language)
    
@app.route("/api/v1/search", methods=["GET"])
@cache.cached()
def get_search_endpoint():
    """
    Search for products based on various filters.

    ---
    parameters:
      - name: query
        in: query
        type: string
        required: false
        description: Search query.
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

    query = request.args.get('query')
    threshold = request.args.get('threshold', default=0.21, type=float)
    max_price = request.args.get('maxPrice', type=int)
    min_price = request.args.get('minPrice', type=int)
    category = request.args.get('category')
    on_sale = request.args.get("onSale", default=False, type=bool)
    brands = request.args.get('brands')
    list_ids = request.args.get('ids')
    language = request.args.get('language', default='en')
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=100, type=int)
    sort_by = request.args.get('sortBy')
    ascending = request.args.get('ascending', default=False, type=bool)
    

    if all([i is None for i in [query, category, on_sale, brands, list_ids]]):
        return make_response(jsonify({'error': 'Missing parameter'}), 400)

    filters = {
        'query': query,
        'threshold': threshold,
        'category': category,
        'min_price': min_price,
        'max_price': max_price,
        'brands': brands,
        'on_sale': on_sale,
        'ids': list_ids
    }

    results = retrieve_filtered_images(images_df, filters, language)
    
        
    available_brands, min_overall_price, max_overall_price = '', 0, 0

    if (results is None) or (len(results)==0):
        return make_response(jsonify({'error': 'Results no found'}), 204)
        
    available_brands = results['brand'].unique().tolist()
    min_overall_price = results['price'].min()
    max_overall_price = results['price'].max()
    all_img_urls = results['img_url'].head(30).unique().tolist()
    good_keys = images_tags_df.index.intersection(all_img_urls)
    all_tags = images_tags_df.loc[good_keys]['tags'].explode().drop_duplicates().map(translations.tags[language]).dropna().unique().tolist()
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

@app.route("/api/v1/product", methods=["GET"])
@cache.cached()
def get_product_endpoint():
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


    result = retrieve_product_data(all_images_df, id, language)
    
    return jsonify({
        'result': result
    })
       

@app.route("/api/v1/most_similar_items", methods=["GET"])
@cache.cached()
def retrieve_most_similar_items_endpoint():
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

    results = retrieve_most_similar_items(id, top_k)
    
    if not results:
        return make_response(jsonify({'error': 'Id not found'}), 400)
        
    return jsonify({
        'results': results
    })

@app.route("/api/v1/image_query_similarity", methods=["GET"])
@cache.cached()
def get_image_query_similarity_endpoint():
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
        return make_response(jsonify({'error': f"{e}"}), 400)
       
    return jsonify({
        'similarity_score': score
    })


if __name__ == "__main__":
    app.run(debug=True)
