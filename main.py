import clip
import pandas as pd
import torch.nn.utils.rnn as r
import cloudpickle as cpickle
import torch
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import numpy as np
from tqdm import tqdm
import pickle
import io
import fetch_data
import json

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


app = Flask(__name__)
CORS(app)

model, preprocess = clip.load("ViT-L/14@336px")
model.cpu().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

blacklist = ['https://www.stradivarius.com/es/coletero-grande-saten-l00150411']

TABLE_ID = 'dokuso.listing_products.items_details'
query = f"SELECT * FROM `{TABLE_ID}`"
images_df = fetch_data.query_datasets_to_df(query).set_index('img_url')

#images_df = pd.read_csv('./data/all_brands.csv', index_col=[0]).set_index('img_url')
blacklist_img_url = images_df[images_df['shop_link'].isin(blacklist)].index.tolist()
print('blacklist_img_url', blacklist_img_url)
images_df = images_df[~images_df['shop_link'].isin(blacklist)]

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


def compute_text_embeddings(text):
    text_tokens = clip.tokenize([text]).cpu()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).cpu().float()
    return text_features

def get_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu() @ image_features.cpu().T
    return similarity

def retrieve_most_similar_images(prompt, max_k=100, threshold = 0.2):
    prompt = prompt.lower().replace('style', '')
    prompt = prompt.replace('fashion', '')
    prompt = prompt.replace('inspiration', '')
    prompt =prompt.lower()
    texts = [template.format(prompt) for template in templates] 
    e_text_cat = torch.cat([compute_text_embeddings(t) for t in texts]).to('cpu')
    
    similarity = get_similarity(e_img_cat, e_text_cat)
    top_idx = np.array(list(set(similarity.topk(max_k).indices.ravel().tolist())))
    top_idx = top_idx[(similarity[:, top_idx].max(0).values>threshold).numpy()]
    
    similar_items = []
    for idx in top_idx:
        idx = (idx.item())
        idx_ = e_img_df.index[idx]
        if idx_ not in blacklist_img_url:
            item_data = {}
            item_data['id'] = images_df.loc[idx_]['id']
            item_data['similarity'] = similarity[:, idx].max().item()
            item_data['brand'] = images_df.loc[idx_]['brand']
            item_data['name'] = images_df.loc[idx_].get('name')
            item_data['section'] = images_df.loc[idx_].get('section')
            item_data['img_url'] = idx_
            item_data['price'] = images_df.loc[idx_].get('price')
            item_data['price_float'] = images_df.loc[idx_].get('price_float')
            item_data['old_price'] = images_df.loc[idx_].get('old_price')
            item_data['old_price_float'] = images_df.loc[idx_].get('old_price_float')
            item_data['discount_rate'] = images_df.loc[idx_].get('discount_rate')
            item_data['sale'] = bool(images_df.loc[idx_].get('sale'))
            item_data['shop_link'] = images_df.loc[idx_].get('shop_link')
            similar_items.append(item_data)
        
    if len(similar_items)>0:
        similar_items = pd.DataFrame(similar_items)
        similar_items = similar_items.sort_values(by='similarity', ascending=False)
        similar_items = similar_items.drop_duplicates(subset=['img_url'], keep='first')
        if max_k is not None:
            similar_items = similar_items.head(max_k)
        similar_items = similar_items.to_dict('records')
   
    return similar_items

@app.route("/api/v1/search", methods=["GET"])
def index():
    query = request.args.get('query')
    section = request.args.get('section')
    on_sale = request.args.get("on_sale", default=False, type=bool)
    brand = request.args.get('brand')
    list_ids = request.args.get('ids')
    # list_ids = request.args.get('ids', type=list)
    print(list_ids)
    
    if all([i is None for i in [query, section, on_sale, brand, list_ids]]):
        make_response(jsonify({'error': 'Missing parameter'}), 400)

    else:
        results = []
        if query is not None:
            query_results = retrieve_most_similar_images(query)
            for item in query_results:
                results.append(item)
        if section is not None:
            filtered = images_df[images_df['section']==section].reset_index().to_dict('records')
            for item in filtered:
                results.append(item)
        if brand is not None:
            filtered = images_df[images_df['brand']==brand].reset_index().to_dict('records')
            for item in filtered:
                results.append(item)
        if on_sale is not False:
            filtered = images_df[images_df['sale']==on_sale].reset_index().to_dict('records')
            for item in filtered:
                results.append(item)
        if list_ids is not None:
            list_ids = list_ids.replace("'", "").split(',')
            print(list_ids)
            list_img_url = images_df[images_df['id'].isin(list_ids)].index.tolist()
            if len(list_img_url)>0:
                for idx_ in list_img_url:
                    print(idx_)
                    item_data = {}
                    item_data['id'] = images_df.loc[idx_]['id']
                    item_data['brand'] = images_df.loc[idx_]['brand']
                    item_data['name'] = images_df.loc[idx_].get('name')
                    item_data['section'] = images_df.loc[idx_].get('section')
                    item_data['img_url'] = idx_
                    item_data['price'] = images_df.loc[idx_].get('price')
                    item_data['price_float'] = images_df.loc[idx_].get('price_float')
                    item_data['old_price'] = images_df.loc[idx_].get('old_price')
                    item_data['old_price_float'] = images_df.loc[idx_].get('old_price_float')
                    item_data['discount_rate'] = images_df.loc[idx_].get('discount_rate')
                    item_data['sale'] = bool(images_df.loc[idx_].get('sale'))
                    item_data['shop_link'] = images_df.loc[idx_].get('shop_link')
                    print(item_data)
                    results.append(item_data)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)