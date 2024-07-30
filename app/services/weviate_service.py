import weaviate
from weaviate.auth import AuthApiKey
from app.services.text_service import TextService
from app.common.exceptions import DataNotFoundError, InvalidParameterError, ServiceUnavailableError, ProcessingError, AuthenticationError, AuthorizationError
from app.config import WEAVIATE_CLASS_NAME, PROPERTIES
import os

class WeaviateService:
    def __init__(self):
        self.client = self.init_client()
        self.text_service = TextService()  # Initialize TextService here
        self.properties = PROPERTIES
        self.weaviate_class_name = WEAVIATE_CLASS_NAME
        
    def init_client(self):
        URL = os.getenv('WEAVIATE_URL')
        APIKEY = os.getenv('WEAVIATE_API_KEY')
        try:
            client = weaviate.Client(
                url=URL,
                auth_client_secret=AuthApiKey(api_key=APIKEY)
            )
        except weaviate.exceptions.AuthenticationFailedException as e:
            raise AuthenticationError("Failed to authenticate with Weaviate")
        except Exception as e:
            raise ServiceUnavailableError("Weaviate service is currently unavailable")
        return client

    def add_data(self, data):
        if not isinstance(data, dict):
            raise InvalidParameterError("Data must be a dictionary")
        self.client.data_object.create(data, self.weaviate_class_name)

    def query_data(self, filters=None, limit=100):
        query_builder = self.client.query.get(self.weaviate_class_name, self.properties)
        if filters:
            query_builder = query_builder.with_where(filters)
        query_builder = query_builder.with_limit(limit)
        try:
            return query_builder.do()
        except Exception as e:
            raise ProcessingError(f"Failed to process query: {str(e)}")

    def search_with_text(self, query_text, threshold, search_type="bm25", filters=None, limit=10, offset=0, sort_by=None, ascending=True):
        query_builder = self.client.query.get(self.weaviate_class_name, self.properties)
        count_query = self.client.query.aggregate(self.weaviate_class_name).with_meta_count()

        if search_type == "bm25":
            query_builder = query_builder.with_bm25(query=query_text)
            count_query = count_query.with_bm25(query=query_text)
        elif search_type == "clip":
            query_vector = self.text_service.get_text_embeddings(query_text)
            vector_search = {
                'vector': query_vector,
                'distance': threshold,
            }
            query_builder = query_builder.with_near_vector(vector_search)
            count_query = count_query.with_near_vector(vector_search)
        elif search_type == "hybrid":
            query_vector = self.text_service.get_text_embeddings(query_text)
            hybrid_search = {
                'query': query_text,
                'vector': query_vector,
                'alpha': 0.25,
                'distance': threshold,
            }
            query_builder = query_builder.hybrid(hybrid_search)
            count_query = count_query.hybrid(hybrid_search)
            
        query_builder = query_builder.\
                            with_limit(limit).\
                            with_offset(offset).\
                            with_additional(['id', 'distance'])

        if filters:
            filter_condition = {
                "operator": "And",
                "operands": filters
            }
            query_builder = query_builder.with_where(filter_condition)
            count_query = count_query.with_where(filter_condition)

        if sort_by:
            order = 'asc' if ascending else 'desc'
            query_builder = query_builder.with_sort({
                'path': [sort_by], 
                'order': order
            })

        try:
            results = query_builder.do()            
            items = results.get('data', {}).get('Get', {}).get(self.weaviate_class_name, [])
            
            # count_result = count_query.do()
            # total_results = count_result['data']['Aggregate'][self.weaviate_class_name][0]['meta']['count']
            total_results = len(items)

            return items, total_results

        except Exception as e:
            raise ProcessingError(f"Failed to process query: {str(e)}")

    def get_product_details(self, product_id, language):
        operands = [
            {"path": ["id_item"], "operator": "Equal", "valueString": product_id},
            {"path": ["language"], "operator": "Equal", "valueString": language}
        ]

        result = self.client.query.get(
            self.weaviate_class_name, self.properties
        ).with_where({
            "operator": "And",
            "operands": operands
        }).do()

        return result

    def get_similar_items(self, product_id, top_k=20, sort_by=None, ascending=True, filters=None):
        product_vector = self.client.query.get(
            self.weaviate_class_name, ["_additional { vector }"]
        ).with_where({
            "path": ["id_item"],
            "operator": "Equal",
            "valueString": product_id
        }).do()
        
        if 'errors' in product_vector:
            raise ProcessingError(f"Query failed: {product_vector['errors']}")
        if not product_vector.get('data', {}).get('Get', {}).get(self.weaviate_class_name):
            raise DataNotFoundError('Product not found or no data returned')

        vector = product_vector['data']['Get'][self.weaviate_class_name][0]['_additional']['vector']

        query_builder = self.client.query.get(
            self.weaviate_class_name, self.properties
        ).with_near_vector({
            "vector": vector
        }).with_limit(top_k)

        if filters:
            query_builder = query_builder.with_where({
                "operator": "And",
                "operands": filters
            })

        if sort_by:
            order = 'asc' if ascending else 'desc'
            query_builder = query_builder.with_sort({
                'path': [sort_by], 
                'order': order}
                )

        return query_builder.do()

    def get_all_brands(self):
        try:
            # Define the filter and group by criteria for entries where country is 'es'
            group_by_filter_criteria = {
                "operator": "Equal",
                "path": ["country"],
                "valueString": "es"
            }

            response = (
                self.client.query
                .aggregate(self.weaviate_class_name)
                .with_group_by_filter(group_by_filter_criteria) 
                .with_meta_count()
                .with_fields("meta { count } groupedBy { path value }")
                .do()
            )
            return [item['value'] for item in response['data']['Aggregate'][self.weaviate_class_name]['groupedBy']]
        except Exception as e:
            raise ProcessingError(f"Failed to retrieve brands: {str(e)}")