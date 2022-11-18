import clip
import pandas as pd
import torch.nn.utils.rnn as r
import cloudpickle as cpickle
import torch
from flask import Flask, jsonify, request, render_template
import os
import requests

app = Flask(__name__,template_folder='templates')

model, preprocess = clip.load("ViT-L/14@336px")
model.cpu().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

with open('../app/mango_embeddings_she_cpu.pkl', 'rb') as f:
    e = cpickle.load(f)
    
all_images_df = pd.read_csv('../app/mango_small_she.csv', index_col=[0])

def get_img(url_image):
    if not pd.isna(url_image):
        response = requests.get(url_image)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img

def compute_image_embeddings(url_image):
    try:
        img = get_img(url_image)
        img = preprocess(img)
        img_tensor = r.pad_sequence([img],  batch_first=True).cuda() #maybe np.stack?
        with torch.no_grad():
            image_features = model.encode_image(img_tensor).float()
        return image_features
    except Exception as e:
        return None

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


def retrieve_most_similar_images(prompt):
    e_t = compute_text_embeddings(prompt)
    similarity_df = pd.DataFrame([{k: get_similarity(e[k], e_t)[0][0] for k in e.keys() if e[k] is not None}]).T.rename(columns={0: 'Similarity'})
    k = 8
    k_items = similarity_df.sort_values(by='Similarity', ascending=False).index[:k]
    iamges_df = all_images_df.loc[k_items]
    return iamges_df[['name', 'img_urls']].to_dict('records')
  
  
@app.route("/search", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt_text = request.form.get("prompt")
        
        results = [v['img_urls'] for v in retrieve_most_similar_images(prompt_text)]

        return render_template("index.html", images=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)