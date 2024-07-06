import os
import pandas as pd
import cloudpickle as pkl
from google.oauth2 import service_account
from google.cloud import bigquery
from app.config.config import CACHE_DIR

class DataAccess:
    def __init__(self):
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise ValueError("Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' not set")
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.bigquery_client = bigquery.Client(credentials=credentials)
        self.bucket_name = 'fashion-embeddings'  # Example bucket name

    def query_datasets_to_df(self, query):
        try:
            query_job = self.bigquery_client.query(query)
            results = query_job.result()
            return results.to_dataframe()
        except Exception as e:
            print(f"Failed to execute query: {e}")
            return pd.DataFrame()

    def store_data_locally(self, data, file_name):
        try:
            file_path = os.path.join(CACHE_DIR, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pkl.dump(data, f)
        except Exception as e:
            print(f"Failed to store data locally: {e}")

    def load_data_locally(self, file_name):
        try:
            file_path = os.path.join(CACHE_DIR, file_name)
            with open(file_path, 'rb') as f:
                data = pkl.load(f)
            return data
        except Exception as e:
            print(f"Failed to load data locally: {e}")
            return None
