import clip
import pandas as pd
import torch.nn.utils.rnn as r
import cloudpickle as cpickle
import torch
from flask import Flask, jsonify, request, make_response
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model, preprocess = clip.load("ViT-L/14@336px")
model.cpu().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

with open('./data/mango_embeddings_small_cpu.pkl', 'rb') as f:
    e_mango = cpickle.load(f)
    
with open('./data/zara_embeddings_small_cpu.pkl', 'rb') as f:
    e_zara = cpickle.load(f)
    
with open('./data/vogue_embeddings.pkl', 'rb') as f:
    e_vogue = cpickle.load(f)
    
mango_df = pd.read_csv('./data/mango_small.csv', index_col=[0])
zara_df = pd.read_csv('./data/zara_small.csv', index_col=[0])
vogue_df = pd.read_csv('./data/vogue.csv', index_col=[0])

BLACKLIST = ['SUDADERA MUÑECO DE NIEVE']

all_data = {
    'mango': {
        'images': mango_df,
        'embeddings': e_mango
        },
    'zara': {
        'images': zara_df,
        'embeddings': e_zara
        },
    'vogue': {
        'images': vogue_df,
        'embeddings': e_vogue
        }
    }


def compute_text_embeddings(text):
    text_tokens = clip.tokenize([text]).cpu()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).cpu().float()
    return text_features

def get_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return similarity

def retrieve_most_similar_images(prompt, max_k=30, threshold = 0.17):
    
    e_t = compute_text_embeddings(prompt)
    
    similar_items = []
    for source, data in all_data.items():
        e = data['embeddings']
        images_df = data['images']
        for k in e:
            if e[k] is not None:
                similarity = get_similarity(e[k], e_t)[0][0]
                if similarity >= threshold:
                    item_data = {}
                    item_data['similarity'] = similarity
                    item_data['brand'] = source
                    item_data['name'] = images_df.loc[k].get('name')
                    item_data['category'] = images_df.loc[k].get('category')
                    item_data['img_urls'] = images_df.loc[k].get('img_urls')
                    
                    if item_data['name'] not in BLACKLIST:
                        similar_items.append(item_data)
                    
    return pd.DataFrame(similar_items).sort_values(by='similarity', ascending=False).head(max_k).to_dict('records')

@app.route("/api/v1/search", methods=["GET"])
def index():
    text = request.args.get('query')
    if text is None:
        make_response(jsonify({'error': 'Missing text parameter'}), 400)
    results = retrieve_most_similar_images(text)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)