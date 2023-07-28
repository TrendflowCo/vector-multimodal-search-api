import pandas as pd
import cloudpickle as pkl
from google.cloud import storage, bigquery

# client
storage_client = storage.Client()

# bucket
bucket = storage_client.bucket('fashion-embeddings')

# blobs
all_images_blob = bucket.blob('all_images.pkl')
all_images_fclip_blob = bucket.blob('all_images_fclip.pkl')
tags_embeddings_blob = bucket.blob('tags_embeddings.pkl')

def load_data_from_blob(blob):
    with blob.open("rb") as f:
        return pkl.load(f)

def save_data_to_blob(blob, data):
    with blob.open("wb") as f:
        pkl.dump(data, f)


# bq functions obtained from https://blog.coupler.io/how-to-crud-bigquery-with-python/

def query_datasets_to_df(QUERY):
    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    client = bigquery.Client()
    
    query_job = client.query(QUERY)
    
    df = query_job.to_dataframe()
    return df