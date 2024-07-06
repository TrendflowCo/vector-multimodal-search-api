import torch
import clip
from joblib import Memory
from app.config.config import CACHE_DIR
import torch.nn.utils.rnn as r
import torch

# Load the CLIP model
model, preprocess = clip.load("ViT-L/14@336px")
model.cpu().eval()

# Create a cache for storing text embeddings
memory = Memory(CACHE_DIR, verbose=0)

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
    return image_features
