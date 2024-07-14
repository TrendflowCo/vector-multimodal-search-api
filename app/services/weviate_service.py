import weaviate
from weaviate.auth import AuthApiKey
from app.services.text_service import TextService
from app.common.exceptions import DataNotFoundError, InvalidParameterError, ServiceUnavailableError, ProcessingError, AuthenticationError, AuthorizationError
from app.config import WEVIATE_CLASS_NAME  # Import the WEVIATE_CLASS_NAME at the top of the file

class WeaviateService:
    def __init__(self):
        self.client = self.init_client()
        self.text_service = TextService()  # Initialize TextService here

    def init_client(self):
        URL = "https://clip-fwrabf88.weaviate.network"
        APIKEY = "ewBgPsrbT93YNHglKOlSlGoOtqF8Vk1UJEpP"
        try:
            client = weaviate.Client(
                url=URL,
                auth_client_secret=AuthApiKey(api_key=APIKEY)
            )
        except weaviate.AuthenticationError as e:
            raise AuthenticationError("Failed to authenticate with Weaviate")
        except weaviate.AuthorizationError as e:
            raise AuthorizationError("Authorization failed for Weaviate access")
        except Exception as e:
            raise ServiceUnavailableError("Weaviate service is currently unavailable")
        return client

    def add_data(self, data):
        if not isinstance(data, dict):
            raise InvalidParameterError("Data must be a dictionary")
        self.client.data_object.create(data, WEVIATE_CLASS_NAME)

    def query_data(self, properties, filters=None, limit=100):
        query_builder = self.client.query.get(WEVIATE_CLASS_NAME, properties)
        if filters:
            query_builder = query_builder.with_where(filters)
        query_builder = query_builder.with_limit(limit)
        try:
            return query_builder.do()
        except Exception as e:
            raise ProcessingError(f"Failed to process query: {str(e)}")

    def search_with_text(self, properties, query_text, threshold, search_type="bm25", filters=None, limit=10, offset=0, sort_by=None, ascending=True):
        query_builder = self.client.query.get(WEVIATE_CLASS_NAME, properties)
        if search_type == "bm25":
            query_builder = query_builder.with_bm25(query=query_text)
        elif search_type == "clip":
            # Using get_text_embeddings from TextService
            query_vector = self.text_service.get_text_embeddings(query_text)
            query_builder = query_builder.with_near_vector({
                'vector': query_vector,
                'distance': threshold,  # max accepted distance
                })
        elif search_type == "hybrid":
            # Using get_text_embeddings from TextService
            query_vector = self.text_service.get_text_embeddings(query_text)
            query_builder = query_builder.hybrid({
                'query': query_text,
                'vector': query_vector,
                'alpha': 0.25,
                'distance': threshold,  # max accepted distance
            })
            
        query_builder = query_builder.\
                            with_limit(limit).\
                            with_offset(offset).\
                            with_additional('distance')

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

        try:
            return query_builder.do()
        except Exception as e:
            raise ProcessingError(f"Failed to process query: {str(e)}")

    def get_product_details(self, properties, product_id, language, sort_by, ascending, filters=None):
        operands = [
            {"path": ["id_item"], "operator": "Equal", "valueString": product_id},
            {"path": ["language"], "operator": "Equal", "valueString": language}
        ]
        if filters:
            operands.extend(filters)

        result = self.client.query.get(
            WEVIATE_CLASS_NAME, properties
        ).with_where({
            "operator": "And",
            "operands": operands
        }).do()

        return result

    def get_similar_items(self, properties, product_id, top_k=20, sort_by=None, ascending=True, filters=None):
        product_vector = self.client.query.get(
            WEVIATE_CLASS_NAME, ["_additional { vector }"]
        ).with_where({
            "path": ["id_item"],
            "operator": "Equal",
            "valueString": product_id
        }).do()
        
        if 'errors' in product_vector:
            raise ProcessingError(f"Query failed: {product_vector['errors']}")
        if not product_vector.get('data', {}).get('Get', {}).get(WEVIATE_CLASS_NAME):
            raise DataNotFoundError('Product not found or no data returned')

        vector = product_vector['data']['Get'][WEVIATE_CLASS_NAME][0]['_additional']['vector']

        query_builder = self.client.query.get(
            WEVIATE_CLASS_NAME, properties
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
        # Use WEVIATE_CLASS_NAME directly
        result = self.client.query.aggregate(
            WEVIATE_CLASS_NAME
        ).with_fields(
            "groupedBy { value }"
        ).with_group_by_filter(
            "brand"
        ).do()
        
        if 'data' in result and 'Aggregate' in result['data']:
            brands = [item['groupedBy']['value'] for item in result['data']['Aggregate'][WEVIATE_CLASS_NAME]]
            return brands
        return []
