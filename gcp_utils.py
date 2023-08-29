import pandas as pd
import cloudpickle as pkl
from google.cloud import storage, bigquery

# client
storage_client = storage.Client()

# bucket
bucket = storage_client.bucket('fashion-embeddings')

# blobs
# all_images_blob = bucket.blob('all_images.pkl')
# all_images_fclip_blob = bucket.blob('all_images_fclip.pkl')
# tags_embeddings_blob = bucket.blob('tags_embeddings.pkl')

def load_data_from_blob(blob):
    with blob.open("rb") as f:
        return pkl.load(f)

def save_data_to_blob(blob, data):
    with blob.open("wb") as f:
        pkl.dump(data, f)

