import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import re
import torch
from googletrans import Translator

def translate(text, source_language):
    translator = Translator()
    translation = translator.translate(text, src=source_language, dest='en')
    return translation.text

def get_img(image_url):
    """
    Fetch an image from a URL and return it as a PIL Image object.

    Args:
        image_url (str): URL of the image.

    Returns:
        PIL.Image: Image object.
    """
    if not pd.isna(image_url):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    return None

def create_item_data(row):
    """
    Create a dictionary of item data from a DataFrame row.

    Args:
        row (pd.Series): A row from a DataFrame.

    Returns:
        dict: A dictionary containing item data.
    """
    item_data = {
        'id': row.get('id', ''),
        'name': row.get('name', ''),
        'price': row.get('price', 0),
        'description': row.get('description', ''),
        'img_url': row.get('img_url', ''),
        'brand': row.get('brand', ''),
        'category': row.get('category', '')
    }
    return item_data

def extract_negative_keywords(query):
    """
    Extract negative keywords from a query string.

    Args:
        query (str): The query string.

    Returns:
        list: A list of negative keywords.
    """
    negative_words = re.findall(r'-\w+', query)
    return [word.strip('-') for word in negative_words]

def paginate_results(results, page, limit):
    """
    Paginate a list of results.

    Args:
        results (list): List of results.
        page (int): Page number.
        limit (int): Number of results per page.

    Returns:
        tuple: Total results, total pages, and paginated results.
    """
    total_results = len(results)
    total_pages = (total_results + limit - 1) // limit
    start = (page - 1) * limit
    end = start + limit
    paginated_results = results[start:end]
    return total_results, total_pages, paginated_results

def concatenate_embeddings(embeddings_list, device='cpu'):
    """
    Concatenate a list of tensor embeddings.
    
    Args:
        embeddings_list (list of torch.Tensor): List of tensor embeddings to concatenate.
        device (str): The device to which the concatenated tensor will be moved ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: Concatenated embeddings tensor.
    """
    # Concatenate all embeddings into a single tensor
    concatenated_embeddings = torch.cat(embeddings_list)
    # Move the concatenated tensor to the specified device
    concatenated_embeddings = concatenated_embeddings.to(device)
    return concatenated_embeddings
