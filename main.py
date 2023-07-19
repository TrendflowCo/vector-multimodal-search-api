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
import fetch_data
from utils import *
from joblib import Memory
import functools
import faiss
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

# Create a cache for storing text embeddings
cache_dir = './cache'
memory = Memory(cache_dir, verbose=0)

app = Flask(__name__)
cache = Cache(app)
CORS(app)

# Set up cache configuration
app.config['CACHE_TYPE'] = 'SimpleCache'
# app.config['CACHE_DIR'] = './app_cache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600*24 # 5 minutes

model, preprocess = clip.load("ViT-L/14@336px")
model.cpu().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size


class CPU_Unpickler(pickle.Unpickler):
    
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        
blacklist = ['https://www.stradivarius.com/es/coletero-grande-saten-l00150411']

TABLE_ID = 'dokuso.listing_products.items_details'

query = f"SELECT * FROM `{TABLE_ID}`"

images_df = fetch_data.query_datasets_to_df(query).set_index('img_url')
images_df['price_float'] = images_df['price_float'].fillna(0)
images_df['old_price_float'] = images_df['old_price_float'].fillna(0)
images_df['discount_rate'] = images_df['discount_rate'].fillna(0)
images_df.loc[images_df['price']=='N/A', 'price'] = '0 $'
images_df.loc[images_df['old_price']=='N/A', 'old_price'] = '0 $'
# del images_df['price_float'], images_df['old_price_float']

blacklist_img_url = images_df[images_df['shop_link'].isin(blacklist)].index.tolist()

with open('./data/all_images.pkl', 'rb') as f:
    # e_img = cpickle.load(f)
    e_img = CPU_Unpickler(f).load()

e_img = {k: v for k,v in e_img.items() if v is not None}
list_urls = list(e_img.keys())

images_df = images_df[images_df.index.isin(list_urls)]

e_img_cat = torch.cat(list(e_img.values()))

e_img_df = pd.DataFrame([], index=e_img.keys())

# Prepare the embeddings for FAISS
embeddings = np.array([embedding.numpy().flatten()for embedding in e_img.values()], dtype=np.float32)

# Build an index with the embeddings
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
index.add(embeddings)

templates = [
    '{}',
    'a photo of a {}',
    'a photo of a {} product',
    'a person wearing {}',
    '{} style',
    '{} fashion',
    '{} inspiration'
]

# Create a dictionary to store image data for faster lookups
image_data_dict = {}
for idx, row in images_df.iterrows():
    image_data_dict[idx] = {
        'id': row['id'],
        'brand': row['brand'],
        'name': row.get('name'),
        'section': row.get('section'),
        'img_url': idx,
        'price': row.get('price'),
        'old_price': row.get('old_price'),
        'discount_rate': row.get('discount_rate'),
        'sale': bool(row.get('sale')),
        'shop_link': row.get('shop_link')
    }

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
    prompt = prompt.lower().replace('style', '')
    prompt = prompt.replace('fashion', '')
    prompt = prompt.replace('inspiration', '')
    prompt = prompt.lower()
    return prompt


def generate_texts(prompt):
    return [template.format(prompt) for template in templates]


def get_image_query_similarity_search(query, img_url):
    texts = generate_texts(query)
    e_text_cat = torch.cat([compute_text_embeddings(t) for t in texts]).to('cpu')
    similarity_score = get_similarity(e_img[img_url], e_text_cat).max().item()
    return similarity_score

def compute_similarities(e_img_cat, query, max_k, threshold, blacklist_img_url):
    query = clean_prompt_text(query)
    texts = generate_texts(query)
    e_text_cat = torch.cat([compute_text_embeddings(t) for t in texts]).to('cpu')
    similarity = get_similarity(e_img_cat, e_text_cat)
    if max_k is not None:
        top_idx = np.array(list(set(similarity.topk(max_k).indices.ravel().tolist())))
    else:
        top_idx = np.arange(similarity.shape[1])  # All indices
    print('Threshold', threshold)
    top_idx = top_idx[(similarity[:, top_idx].max(0).values > threshold).numpy()]

    similar_items = []

    for idx in top_idx:
        idx = idx.item()
        idx_ = e_img_df.index[idx]
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
        'section': row.get('section'),
        'img_url': row.get('img_url'),
        'price': row.get('price'),
        'old_price': row.get('old_price'),
        'price_float': row.get('price_float'),
        'old_price_float': row.get('old_price_float'),
        'discount_rate': row.get('discount_rate'),
        'sale': bool(row.get('sale')),
        'shop_link': row.get('shop_link')
    }

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
        similar_items = compute_similarities(e_img_cat, query, max_k=None, threshold=threshold, blacklist_img_url=None)
        similar_items_df = pd.DataFrame(similar_items).set_index('img_url')
        similar_items_df = similar_items_df.sort_values(by='similarity', ascending=False)
        filtered_items = filtered_items.loc[similar_items_df.index]

    if filters.get('section'):
        filtered_items = filtered_items[filtered_items['section'] == filters['section']]
    if filters.get('min_price'):
        filtered_items = filtered_items[filtered_items['price_float'] >= filters['min_price']]
    if filters.get('max_price'):
        filtered_items = filtered_items[filtered_items['price_float'] <= filters['max_price']]
    if filters.get('brands'):
        list_brands = [x.lower() for x in filters['brands'].replace("'", "").split(',')]
        filtered_items = filtered_items[filtered_items['brand'].str.lower().isin(list_brands)]
    if filters.get('on_sale'):
        filtered_items = filtered_items[filtered_items['sale'] == filters['on_sale']]
    if filters.get('ids'):
        list_ids = filters['ids'].replace("'", "").split(',')
        filtered_items = filtered_items[filtered_items['id'].isin(list_ids)]

    filtered_items.reset_index(inplace=True)
    return filtered_items

def retrieve_most_similar_items(input_id, k):
    
    input_index = images_df[images_df['id'] == input_id].index

    if len(input_index)==0:
        return None
    
    input_img_url = input_index[0]
        
    input_shop_link =  images_df.loc[input_img_url]['shop_link']
    
    input_embedding = e_img[input_img_url].numpy().flatten()
    
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
                similar_images.append(create_item_data(row))
        else:
  
            similar_images.append(create_item_data(row))
    
    return similar_images

def paginate_results(results, page, limit):
    total_results = len(results)
    total_pages = (total_results - 1) // limit + 1
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_results = results[start_index:end_index]

    return total_results, total_pages, paginated_results


@app.route("/api/v1/search", methods=["GET"])
@cache.cached()
def get_search_endpoint():
    query = request.args.get('query')
    threshold = request.args.get('threshold', default=0.23, type=float)
    max_price = request.args.get('maxPrice', type=int)
    min_price = request.args.get('minPrice', type=int)
    section = request.args.get('section')
    on_sale = request.args.get("onSale", default=False, type=bool)
    brands = request.args.get('brands')
    list_ids = request.args.get('ids')
    language = request.args.get('language', default='en')
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=100, type=int)
    sort_by = request.args.get('sortBy')
    ascending = request.args.get('ascending', default=False, type=bool)
    

    if all([i is None for i in [query, section, on_sale, brands, list_ids]]):
        return make_response(jsonify({'error': 'Missing parameter'}), 400)

    filters = {
        'query': query,
        'threshold': threshold,
        'section': section,
        'min_price': min_price,
        'max_price': max_price,
        'brands': brands,
        'on_sale': on_sale,
        'ids': list_ids
    }

    results = retrieve_filtered_images(images_df, filters, language)
    
    if sort_by:
        list_sort_by = sort_by.replace("'", "").split(',')
        results = results.sort_values(by=list_sort_by, ascending=ascending)
        
    available_brands, min_overall_price, max_overall_price = '', 0, 0
    if (results is not None) or (len(results)>0):
        available_brands = results['brand'].unique().tolist()
        min_overall_price = results['price_float'].min()
        max_overall_price = results['price_float'].max()
        
    results = results.apply(create_item_data, axis=1)
        
    results = results.tolist()

    total_results, total_pages, paginated_results = paginate_results(results, page, limit)

    return jsonify({
        'results': paginated_results,
        'page': page,
        'limit': limit,
        'total_results': total_results,
        'total_pages': total_pages,
        'metadata': {'brands': available_brands, 
                     'min_price': min_overall_price,
                     'max_price': max_overall_price}
    })
       

@app.route("/api/v1/similarity", methods=["GET"])
@cache.cached()
def retrieve_most_similar_items_endpoint():
    id = request.args.get('id', type=str)
    query = request.args.get('query')

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
def get_item_query_similarity_endpoint():
    query = request.args.get('id', type=str)
    img_url = request.args.get('img_url', type=str)
    

    if all([i is None for i in [id, top_k]]):
        return make_response(jsonify({'error': 'Missing parameter'}), 400)

    score = get_image_query_similarity_search(query, img_url)
    
    if not results:
        return make_response(jsonify({'error': 'Id not found'}), 400)
       
    return jsonify({
        'similarity_score': score
    })
    
if __name__ == "__main__":
    app.run(debug=True)
