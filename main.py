import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as r
import clip
from flask import Flask, jsonify, request, make_response
from flask_caching import Cache
from flask_cors import CORS
from joblib import Memory
import functools
import faiss
from utils import *
from fetch_data import *
from templates import templates, templates_with_adjectives, garment_types
from flasgger import Swagger
import translations
import requests
from PIL import Image
from io import BytesIO
import re

TAGS_THRESHOLD = 0.24

# Load the CLIP model
model, preprocess = clip.load("ViT-L/14@336px")
model.cpu().eval()
        

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
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600*24 # 5 minutes
app.config['SWAGGER'] = {
    'title': 'Fashion CLIP Search API',
    'uiversion': 3
}


combined_query = """
SELECT *
FROM `dokuso.production.combined`
"""

combined_df = query_datasets_to_df(combined_query).drop_duplicates(subset='shop_link').reset_index(drop=True)

combined_tags_query = """
SELECT *
FROM `dokuso.production.combined_tags`
where value is not null
"""

images_tags_df = query_datasets_to_df(combined_tags_query)
print(len(images_tags_df))

# Extracting all_images_df
all_images_df = combined_df.drop(columns=['features'])
all_images_df['old_price'] = all_images_df['old_price'].fillna(all_images_df['price'])

# Extracting images_tags_df
images_tags_df = images_tags_df[['img_url', 'value']].dropna().drop_duplicates().set_index('img_url')

# Filtering out entries without features
filtered_df = combined_df[~combined_df['features'].isnull()]
del combined_df

# Extracting e_img_clip
e_img_clip = {row['img_url']: torch.tensor(row['features']).reshape(1, -1).float() for _, row in filtered_df.iterrows()}

# Extracting images_df
images_df = all_images_df[all_images_df['img_url'].isin(e_img_clip.keys())].set_index('img_url')

# Extracting e_img_clip_cat
e_img_clip_cat = torch.cat(list(e_img_clip.values()))

# Extracting e_img_clip_df
e_img_clip_df = pd.DataFrame([], index=e_img_clip.keys())

sale_wording = ['on sale', 'sales', 'sale', 'clearance']
all_brands = images_df['brand'].str.lower().unique().tolist()
all_categories = images_df['category'].str.lower().unique().tolist()
conn_brand_words = ['by', 'from', 'of']
conn_cat_words = ['for']
conn_price_words = ['under', 'below', 'for less than', 'less than', 'less']
curr_symbols = ['$', '€', 'dollars', 'euros', 'dollar', 'euro', 'usd', 'eur', 'us$']

e_img_clip_cat = torch.cat(list(e_img_clip.values()))
e_img_clip_df = pd.DataFrame([], index=e_img_clip.keys())

# Prepare the embeddings for FAISS
embeddings = np.array([embedding.numpy().flatten() for embedding in e_img_clip.values()], dtype=np.float32)

# Build an index with the embeddings
index_faiss = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
index_faiss.add(embeddings)

def get_img(image_url):
    """
    Get image from URL.

    Args:
        image_url (str): URL of the image.
      
    Returns:
        img (PIL.Image): Image.
    """
    if not pd.isna(image_url):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img

@memory.cache
def compute_text_embeddings(text):
    """
    Compute text embeddings.
    
    Args:
        text (str): Text to embed.
    
    Returns:
        text_features (torch.tensor): Text embeddings.
    """
    text_tokens = clip.tokenize([text]).cpu()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).cpu().float()
    return text_features

@memory.cache
def compute_image_embeddings(image_url):
    """
    Compute image embeddings.
    
    Args:
        image_url (str): URL of the image.
    
    Returns:
        image_features (torch.tensor): Image embeddings.
    """
    img = get_img(image_url)
    img = preprocess(img)
    img_tensor = r.pad_sequence([img],  batch_first=True).cpu() #maybe np.stack?
    with torch.no_grad():
        image_features = model.encode_image(img_tensor).float()
        # image_features = model.encode_image(img_tensor).reshape(1, -1).float()
    return image_features


@functools.lru_cache(maxsize=None)
def get_similarity(image_features, query_features):
    image_features_normalized = F.normalize(image_features, dim=1)
    query_features_normalized = F.normalize(query_features, dim=1)
    similarity = query_features_normalized @ image_features_normalized.T
    return similarity


# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# def encode(texts):
    
#     try:
#         # Tokenize sentences
#         encoded_input = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')

#         # Compute token embeddings
#         with torch.no_grad():
#             model_output = model_miniLM(**encoded_input)

#         # Perform pooling
#         sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

#         # Normalize embeddings
#         sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

#         return sentence_embeddings
    
#     except:
#         return None
    

# def retrieve_most_similar_text(query, THRESHOLD = 0.35):
#     question = [query]
#     output = encode(question)
#     query_embeddings = torch.FloatTensor(output)

#     hits = semantic_search(query_embeddings, e_text_fclip_cat, top_k=len(items_df))
#     results_df = pd.DataFrame(hits[0])
#     if len(hits[0]) == 0:
#         raise ValueError('Error')
#     ids = results_df['corpus_id'].tolist()
#     results = items_df.loc[[list_text_item_ids[id_] for id_ in ids]]
#     results['query_score'] =  results_df['score'].tolist()
#     results = results[results['query_score']>=THRESHOLD]

#     return results.reset_index()
    
def clean_prompt_text(prompt):
    """
    Clean prompt text.
    
    Args:
        prompt (str): Prompt text.
    
    Returns:
        prompt (str): Cleaned prompt text.
    """
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
    """
    Generate texts from prompt and templates.
    
    Args:
        prompt (str): Prompt text.
        templates (list): List of templates.
    
    Returns:
        texts (list): List of generated texts.
    """
    if any([x in garment_types for x in prompt.split(' ')]):
        return [template.format(prompt) for template in templates]
    else:
        return [template.format(prompt) for template in templates_with_adjectives]


def get_image_query_similarity_score(query, img_url):
    """
    Get similarity score between an image query and an image URL.
    
    Args:
        query (str): Image query.
        img_url (str): Image URL.
    
    Returns:
        similarity_score (float): Similarity score.
        error (str): Error message.
    """
    try:
        query = clean_prompt_text(query)
        texts = generate_texts(query, templates)
        e_text_cat = torch.cat([compute_text_embeddings(t) for t in texts]).to('cpu')
        similarity_score = get_similarity(e_img_clip[img_url], e_text_cat).max().item()
        return similarity_score, None
    except Exception as e:
        return None, str(e)

def compute_text_image_similarities(e_img_clip_cat, query, max_k, threshold):
    """
    Compute similarities between text and images.
    
    Args:
        e_img_clip_cat (torch.tensor): Image embeddings.
        query (str): Text query.
        max_k (int): Maximum number of similar items to retrieve.
        threshold (float): Minimum similarity score.
    
    
    Returns:
        similar_items (list): List of similar items.
    """
    query = clean_prompt_text(query)
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

def compute_image_image_similarities(e_img_clip_cat, image_url, max_k, threshold):
    """
    Compute similarities between images.
    
    Args:
        e_img_clip_cat (torch.tensor): Image embeddings.
        image_url (str): Image URL.
        max_k (int): Maximum number of similar items to retrieve.
        threshold (float): Minimum similarity score.
    
    
    Returns:
        similar_items (list): List of similar items.
    """
    e_query_img = compute_image_embeddings(image_url)
    similarity = get_similarity(e_img_clip_cat, e_query_img)
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
    """
    Create item data.
    
    Args:
        row (pd.Series): Row of dataframe.
    
    
    Returns:
        item_data (dict): Item data.
    """
    return {
        'id': row['id'],
        'brand': row['brand'],
        'name': row.get('name'),
        'category': row.get('category'),
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


def retrieve_filtered_images(images_df, filters):
    """
    Retrieve filtered images.
    
    Args:
        images_df (pd.DataFrame): Dataframe of images.
        filters (dict): Filters.
    
    Returns:
        filtered_items (pd.DataFrame): Filtered images.
    """
    filtered_items = images_df.copy()

    if filters.get('query'):
        
        if filters['language'] != 'en':
          print(f"Translation from {filters['language']} to en: \n")
          print(f"Original: {filters['query']}\n")
          filters['query'] = translate(filters['query'], filters['language'])
          print(f"English: {filters['query']}")
        threshold = filters.get('threshold', 0.21)
        
        if filters['image_url']:
            similar_items = compute_image_image_similarities(e_img_clip_cat, filters['query'], max_k=None, threshold=threshold)
        else:
            filters['query'] = filters['query'].lower()
            
            if any([w in filters['query'] for w in sale_wording]):
                filters['on_sale'] = True
                for w in sale_wording:
                    filters['query'] = filters['query'].replace(w, '')
            query_brands = [brand for brand in all_brands if brand in filters['query']]
            
            if len(query_brands)>0:
                filters['brands'] = ','.join(query_brands)
                for brand in query_brands:
                    filters['query'] = filters['query'].replace(brand+"'s", '').replace(brand, '')
                for w in conn_brand_words:
                    filters['query'] = filters['query'].replace(w, '')

            query_categories = []
            for w in filters['query'].split(' '):
                w_ = w.replace('s', '').replace("'", '')
                if w_ in all_categories:
                    query_categories.append(w_)
            
            if len(query_categories)>0:
                filters['category'] = query_categories[0]
                for cat in query_categories:
                    filters['query'] = filters['query'].replace(cat+"'s", '').replace(cat, '')
                for w in conn_cat_words:
                    filters['query'] = filters['query'].replace(w, '')
            
            if any([w in filters['query'] for w in curr_symbols]):
                price_pattern = r'(\d+(?:\.\d{1,2})?)\s*(?:dollars|dollar|euros|euro|\$|€)'
                price_match = re.search(price_pattern, filters['query'], re.IGNORECASE)
                if price_match:
                    # Extract the matched price as a float
                    filters['max_price'] = float(price_match.group(1))
                    for curr in curr_symbols:
                        filters['query'] = filters['query'].replace(curr, '')
                    for w in conn_price_words:
                        filters['query'] = filters['query'].replace(w, '')
            

            negative_keywords = extract_negative_keywords(filters['query'])
            clean_query = ' '.join([w for w in filters['query'].split(' ') if w.replace('-', '') not in negative_keywords])
            similar_items = compute_text_image_similarities(e_img_clip_cat, clean_query, max_k=None, threshold=threshold)
       
        if len(similar_items) == 0:
            return None
        
        similar_items_df = pd.DataFrame(similar_items).set_index('img_url')

        if len(negative_keywords)>0:
            blacklist = []
            for keyword in negative_keywords:
                negative_results = compute_text_image_similarities(e_img_clip_cat, f'{clean_query} {keyword}', max_k=None, threshold=threshold)
                blacklist += [item['img_url'] for item in negative_results]
            similar_items_df = similar_items_df[~similar_items_df.index.isin(blacklist)]

        similar_items_df = similar_items_df.sort_values(by='similarity', ascending=False)
        filtered_items = filtered_items.loc[similar_items_df.index]
        filtered_items['similarity'] = similar_items_df['similarity']
    
    if filters.get('category'):
        filtered_items = filtered_items[filtered_items['category'] == filters['category']]
    
    if filters.get('min_price'):
        filtered_items = filtered_items[filtered_items['price'] >= filters['min_price']]
    
    if filters.get('max_price'):
        filtered_items = filtered_items[filtered_items['price'] <= filters['max_price']]
    
    if filters.get('tags'):
        list_tags = [x.lower() for x in filters['tags'].replace("'", "").split(',')]
        images_tags_filtered_df = images_tags_df[images_tags_df['value'].str.lower().isin(list_tags)]
        list_tags_img_ids = images_tags_filtered_df['img_id'].unique().tolist()
        filtered_items = filtered_items[filtered_items['img_id'].isin(list_tags_img_ids)]
    
    if filters.get('brands'):
        list_brands = [x.lower() for x in filters['brands'].replace("'", "").split(',')]
        filtered_items = filtered_items[filtered_items['brand'].str.lower().isin(list_brands)]
    
    if filters.get('on_sale'):
        filtered_items = filtered_items[filtered_items['sale'] == filters['on_sale']]
    
    if filters.get('ids'):
        list_ids = filters['ids'].replace("'", "").split(',')
        filtered_items = filtered_items[filtered_items['id'].isin(list_ids)]
    
    return filtered_items.reset_index().drop_duplicates(['img_url', 'shop_link'])

def retrieve_most_similar_items(input_id, k):
    """
    Retrieve most similar items to a given product.
    
    Args:
        input_id (str): ID of the product.
        k (int): Number of similar items to retrieve.
    
    Returns:
        similar_images (list): List of similar items.
    """
    input_index = images_df[images_df['id'] == input_id].index
    if len(input_index)==0:
        return None

    input_img_url = input_index[0]
    input_shop_link =  images_df.loc[input_img_url]['shop_link']
    input_embedding = e_img_clip[input_img_url].numpy().flatten()
    similar_images = faiss_search(k, input_shop_link, input_embedding)
    
    return similar_images

def faiss_search(k, input_shop_link, input_embedding):
    """
    Perform FAISS search.
    
    Args:
        k (int): Number of similar items to retrieve.
        input_shop_link (str): Shop link of the input product.
        input_embedding (np.array): Embedding of the input product.
    
    Returns:
        similar_images (list): List of similar items.
    """
    _, indices = index_faiss.search(np.array([input_embedding], dtype=np.float32), k + 1)

    similar_images = []
    shop_link_similarities = {}
    for i, idx in enumerate(indices[0][1:]):
        # img_url = list_urls[idx]
        img_url = e_img_clip_df.index[idx]
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
    """
    Paginate results.
    
    Args:
        results (list): List of results.
        page (int): Page number.
        limit (int): Number of results per page.
    
    Returns:
        total_results (int): Total number of results.
        total_pages (int): Total number of pages.
        paginated_results (list): List of paginated results.
    """
    total_results = len(results)
    total_pages = (total_results - 1) // limit + 1
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_results = results[start_index:end_index]

    return total_results, total_pages, paginated_results


def create_specific_item_data(item_data, language='en'):
    """
    Create item data.
    
    Args:
        item_data (pd.Series): Row of dataframe.
        language (str): Language for tags.
    
    Returns:
        item_data (dict): Item data.
    """
    good_image_ids = images_tags_df.index.intersection(item_data['img_url'].tolist())
    all_tags = images_tags_df.loc[good_image_ids]['value'].explode().drop_duplicates().map(translations.tags[language]).dropna().unique().tolist()
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
    """
    Retrieve product data.
    
    Args:
        images_df (pd.DataFrame): Dataframe of images.
        id (str): ID of the product.
        language (str): Language for tags.
    
    Returns:
        item_data (dict): Product data.    
    """
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

    results = retrieve_filtered_images(images_df, filters)

    available_brands, min_overall_price, max_overall_price = '', 0, 0

    if (results is None) or (len(results)==0):
        return make_response(jsonify({'error': 'Results no found'}), 204)
        
    available_brands = results['brand'].unique().tolist()
    min_overall_price = results['price'].min()
    max_overall_price = results['price'].max()
    all_img_urls = results['img_url'].head(30).unique().tolist()
    good_keys = images_tags_df.index.intersection(all_img_urls)
    all_tags = images_tags_df.loc[good_keys]['value'].explode().drop_duplicates().map(translations.tags[language]).dropna().unique().tolist()
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

@app.route("/api/v1/brands_list", methods=["GET"])
@cache.cached()
def get_brands_list():
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
       
    return jsonify({
        'brand_list': images_df.brand.unique().tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)
