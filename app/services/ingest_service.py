import pandas as pd
import ast
from google.cloud import bigquery
from app.services.weviate_service import WeaviateService

class IngestService:
    def __init__(self):
        self.weaviate_service = WeaviateService()
        self.bq_client = bigquery.Client()

    def parse_img_urls(self, img_urls):
        if isinstance(img_urls, list):
            return img_urls
        if isinstance(img_urls, str):
            try:
                cleaned_str = img_urls.replace('\n', ',').replace(' ', '')
                return ast.literal_eval(cleaned_str)
            except:
                print(f"Warning: Unable to parse: {img_urls}")
                return []
        else:
            print(f"Warning: Unexpected type for img_urls: {type(img_urls)}")
            return []

    def ingest_data(self):
        # Query for combined data
        query_combined = """
            SELECT 
                shop_link,
                ARRAY_AGG(DISTINCT img_url) AS img_urls,
                ANY_VALUE(id) AS id,
                ANY_VALUE(brand) AS brand,
                ANY_VALUE(category) AS category,
                ANY_VALUE(name) AS name,
                ANY_VALUE(desc_1) AS desc_1,
                ANY_VALUE(desc_2) AS desc_2,
                ANY_VALUE(price) AS price,
                ANY_VALUE(old_price) AS old_price,
                ANY_VALUE(discount_rate) AS discount_rate,
                ANY_VALUE(features) AS features
            FROM 
                `dokuso.production.combined`
            GROUP BY 
                shop_link
        """
        results = self.bq_client.query(query_combined).to_dataframe()

        # Query for tags
        query_tags = """
            SELECT 
                img_url,
                ARRAY_AGG(DISTINCT value) AS values
            FROM 
                `dokuso.production.images_tags`
            WHERE 
                score > 0.18
            GROUP BY 
                img_url
        """
        tags = self.bq_client.query(query_tags).to_dataframe()

        # Process img_urls
        results['img_urls_parsed'] = results['img_urls'].apply(self.parse_img_urls)

        # Create tags dictionary
        tags_dict = {row['img_url']: row['values'] for _, row in tags.iterrows()}

        # Add tags to results
        results['tags'] = results['img_urls_parsed'].apply(lambda x: list(set().union(*[tags_dict.get(url, []) for url in x])))

        # Fill NaN values
        results['category'] = results['category'].fillna('')
        results['desc_1'] = results['desc_1'].fillna('')
        results['desc_2'] = results['desc_2'].fillna('')
        results['sale'] = results['discount_rate'] > 0

        # Prepare data for Weaviate
        data_objects = []
        for _, row in results.iterrows():
            obj = {
                "shop_link": row['shop_link'],
                "img_urls": row['img_urls_parsed'],
                "id": row['id'],
                "brand": row['brand'],
                "category": row['category'],
                "name": row['name'],
                "desc_1": row['desc_1'],
                "desc_2": row['desc_2'],
                "price": float(row['price']),
                "old_price": float(row['old_price']),
                "discount_rate": float(row['discount_rate']),
                "features": ast.literal_eval(row['features']),
                "tags": row['tags'],
                "sale": bool(row['sale'])
            }
            data_objects.append(obj)

        # Ingest data into Weaviate
        self.weaviate_service.batch_import_data(data_objects)

        return len(data_objects)
