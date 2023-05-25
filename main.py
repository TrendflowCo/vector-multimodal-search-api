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
from utils import translate
from joblib import Memory
import functools


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

blacklist_img_url = images_df[images_df['shop_link'].isin(blacklist)].index.tolist()

with open('./data/all_images.pkl', 'rb') as f:
    # e_img = cpickle.load(f)
    e_img = CPU_Unpickler(f).load()
    
e_img = {k: v for k,v in e_img.items() if v is not None}

e_img_cat = torch.cat(list(e_img.values()))

e_img_df = pd.DataFrame(e_img_cat, index=e_img.keys())


templates = [
    '{}.',
    # 'a photo of a {}.',
    'a person wearing {}.',
    '{} style.',
    '{} fashion.',
    '{} inspiration.'
]


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
        'price_float': row.get('price_float'),
        'old_price': row.get('old_price'),
        'old_price_float': row.get('old_price_float'),
        'discount_rate': row.get('discount_rate'),
        'sale': bool(row.get('sale')),
        'shop_link': row.get('shop_link')
    }
    
def compute_similarity(e_img_cat, e_text_cat, max_k, threshold, blacklist_img_url):
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
        
#     similar_items = []

#     for idx in top_idx:
#         idx = idx.item()
#         idx_ = e_img_df.index[idx]
#         if idx_ not in blacklist_img_url:
#             item_data = create_item_data(idx_)
#             item_data['similarity'] = similarity[:, idx].max().item()
#             similar_items.append(item_data)

    return similar_items


# def create_item_data(idx):
#     item_data = {
#         'id': images_df.loc[idx]['id'],
#         'brand': images_df.loc[idx]['brand'],
#         'name': images_df.loc[idx].get('name'),
#         'section': images_df.loc[idx].get('section'),
#         'img_url': idx,
#         'price': images_df.loc[idx].get('price'),
#         'price_float': images_df.loc[idx].get('price_float'),
#         'old_price': images_df.loc[idx].get('old_price'),
#         'old_price_float': images_df.loc[idx].get('old_price_float'),
#         'discount_rate': images_df.loc[idx].get('discount_rate'),
#         'sale': bool(images_df.loc[idx].get('sale')),
#         'shop_link': images_df.loc[idx].get('shop_link')
#     }
#     return item_data


def retrieve_most_similar_images(prompt, max_k=None, threshold=0.2):
    prompt = clean_prompt_text(prompt)
    texts = generate_texts(prompt)
    e_text_cat = torch.cat([compute_text_embeddings(t) for t in texts]).to('cpu')
    similar_items = compute_similarity(e_img_cat, e_text_cat, max_k, threshold, blacklist_img_url)

    if len(similar_items) > 0:
        similar_items = pd.DataFrame(similar_items)
        similar_items = similar_items.sort_values(by='similarity', ascending=False)
        similar_items = similar_items.drop_duplicates(subset=['img_url'], keep='first')
        if max_k is not None:
            similar_items = similar_items.head(max_k)
        similar_items = similar_items.to_dict('records')

    return similar_items

def handle_query_results(query, threshold, language, results):
    if query is not None:
        if (language is not None) and (language != 'en'):
            print(f"Translation from {language} to en: \n")
            print(f"Original: {query}\n")
            query = translate(query, language)
            print(f"English: {query}")

        query_results = retrieve_most_similar_images(query, threshold=threshold)
        results.extend(query_results)

def handle_section_results(section, results):
    if section is not None:
        filtered = images_df[images_df['section'] == section].reset_index().to_dict('records')
        results.extend(filtered)

def handle_brand_results(brand, results):
    if brand is not None:
        filtered = images_df[images_df['brand'] == brand].reset_index().to_dict('records')
        results.extend(filtered)

def handle_on_sale_results(on_sale, results):
    if on_sale is not False:
        filtered = images_df[images_df['sale'] == on_sale].reset_index().to_dict('records')
        results.extend(filtered)

# def handle_list_ids_results(list_ids, results):
#     if list_ids is not None:
#         list_ids = list_ids.replace("'", "").split(',')
#         list_img_url = images_df[images_df['id'].isin(list_ids)].index.tolist()
#         for idx_ in list_img_url:
#             item_data = {}
#             # Construct item_data
#             results.append(item_data) 
        
def handle_list_ids_results(list_ids, images_df):
    results = []
    if list_ids is not None:
        list_ids = list_ids.replace("'", "").split(',')
        filtered_items = images_df[images_df['id'].isin(list_ids)]
        results = filtered_items.apply(create_item_data, axis=1).tolist()
    return results

def paginate_results(results, page, limit):
    total_results = len(results)
    total_pages = (total_results - 1) // limit + 1
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_results = results[start_index:end_index]
    
    return total_results, total_pages, paginated_results

@app.route("/api/v1/search", methods=["GET"])
@cache.cached()
def index():
    query = request.args.get('query')
    threshold = request.args.get('threshold', default=0.25, type=float)
    section = request.args.get('section')
    on_sale = request.args.get("on_sale", default=False, type=bool)
    brand = request.args.get('brand')
    list_ids = request.args.get('ids')
    language = request.args.get('language')
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=100, type=int)

    if all([i is None for i in [query, section, on_sale, brand, list_ids]]):
        return make_response(jsonify({'error': 'Missing parameter'}), 400)

    results = []
    handle_query_results(query, threshold, language, results)
    handle_section_results(section, results)
    handle_brand_results(brand, results)
    handle_on_sale_results(on_sale, results)
    handle_list_ids_results(list_ids, results)


    total_results, total_pages, paginated_results = paginate_results(results, page, limit)

    return jsonify({
        'results': paginated_results,
        'page': page,
        'limit': limit,
        'total_results': total_results,
        'total_pages': total_pages
    })


if __name__ == "__main__":
    app.run(debug=True)
