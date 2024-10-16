import weaviate
from weaviate.auth import AuthApiKey
from app.services.text_service import TextService
from app.common.exceptions import DataNotFoundError, InvalidParameterError, ServiceUnavailableError, ProcessingError, AuthenticationError, AuthorizationError
from app.config import WEAVIATE_CLASS_NAME_ITEMS, WEAVIATE_CLASS_NAME_IMAGES, PROPERTIES, WEAVIATE_URL, WEAVIATE_API_KEY

class WeaviateService:
    def __init__(self):
        self.client = self.init_client()
        self.text_service = TextService()  # Initialize TextService here
        self.properties = PROPERTIES
        self.items_class = WEAVIATE_CLASS_NAME_ITEMS
        self.images_class = WEAVIATE_CLASS_NAME_IMAGES
        
    def init_client(self):
        try:
            client = weaviate.Client(
                url=WEAVIATE_URL,
                auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY)
            )
        except weaviate.exceptions.AuthenticationFailedError as e:
            raise AuthenticationError("Failed to authenticate with Weaviate: " + str(e))
        except Exception as e:
            raise ServiceUnavailableError("Weaviate service is currently unavailable: " + str(e))
        return client

    def add_data(self, data):
        if not isinstance(data, dict):
            raise InvalidParameterError("Data must be a dictionary")
        try:
            self.client.data_object.create(data, self.items_class)
        except weaviate.exceptions.ObjectAlreadyExistsError as e:
            raise ProcessingError("Data object already exists: " + str(e))
        except Exception as e:
            raise ProcessingError("Failed to add data: " + str(e))

    def query_data(self, filters=None, limit=100):
        query_builder = self.client.query.get(self.items_class, self.properties)
        if filters:
            query_builder = query_builder.with_where(filters)
        query_builder = query_builder.with_limit(limit)
        try:
            result = query_builder.do()
            if result is None:
                return []
            return result.get('data', {}).get('Get', {}).get(self.items_class, [])
        except weaviate.exceptions.WeaviateQueryError as e:
            raise ProcessingError("Query failed: " + str(e))
        except Exception as e:
            raise ProcessingError("Failed to process query: " + str(e))

    def search_with_text(self, query_text, threshold, search_type="bm25", filters=None, limit=10, offset=0, sort_by=None, ascending=True):
        query_builder = self.client.query.get(self.items_class, self.properties)
        # count_query = self.client.query.aggregate(self.items_class).with_meta_count()

        if search_type == "bm25":
            query_builder = query_builder.with_bm25(query=query_text)
            # count_query = count_query.with_bm25(query=query_text)
        elif search_type == "clip":
            query_vector = self.text_service.get_text_embeddings(query_text)
            vector_search = {
                'vector': query_vector,
                'distance': threshold,
            }
            query_builder = query_builder.with_near_vector(vector_search)

        elif search_type == "hybrid":
            query_vector = self.text_service.get_text_embeddings(query_text)
            hybrid_search = {
                'query': query_text,
                'vector': query_vector,
                'alpha': 0.25,
                'distance': threshold,
            }
            query_builder = query_builder.hybrid(hybrid_search)
            
        if filters:
            filter_condition = {
                "operator": "And",
                "operands": filters
            }
            query_builder = query_builder.with_where(filter_condition)

        query_builder = query_builder.\
                            with_limit(limit).\
                            with_offset(offset).\
                            with_additional(['id', 'distance'])

        if sort_by:
            order = 'asc' if ascending else 'desc'
            query_builder = query_builder.with_sort({
                'path': [sort_by], 
                'order': order
            })

        try:
            results = query_builder.do()
            if results is None:
                return [], 0
            
            items = results.get('data', {}).get('Get', {}).get(self.items_class, [])
            total_results = len(items)

            return items, total_results

        except Exception as e:
            raise ProcessingError(f"Failed to process query: {str(e)}")

    def get_product_details(self, product_id, filters=None):
        # Query Clipfeatures_items
        
        items_query = (
            self.client.query.get(self.items_class, self.properties)
            .with_where({
                "operator": "And",
                "operands": [
                    {"path": ["id_item"], "operator": "Equal", "valueString": product_id},
                    *filters
                ] if filters else [
                    {"path": ["id_item"], "operator": "Equal", "valueString": product_id}
                ]
            })
        )
        try:
            items_result = items_query.do()
            if items_result is None:
                return None

            # Combine results
            product_details = items_result.get('data', {}).get('Get', {}).get(self.items_class, [])
            if product_details:
                product_details = product_details[0]
                product_details['img_urls'] = product_details['img_urls']
                # product_details['images'] = images_result.get('data', {}).get('Get', {}).get(self.images_class, [])
            return product_details
        except weaviate.exceptions.WeaviateTimeoutError as e:
            raise ServiceUnavailableError("Request to Weaviate timed out: " + str(e))
        except Exception as e:
            raise ProcessingError(f"Failed to retrieve product details: {str(e)}")

    def get_similar_items(self, product_id, top_k=20, sort_by=None, ascending=True, filters=None):
        product_vector = self.client.query.get(
            self.items_class, ["_additional { vector }"]
        ).with_where({
            "path": ["id_item"],
            "operator": "Equal",
            "valueString": product_id
        }).do()
        
        if 'errors' in product_vector:
            raise ProcessingError(f"Query failed: {product_vector['errors']}")
        if not product_vector.get('data', {}).get('Get', {}).get(self.items_class):
            raise DataNotFoundError('Product not found or no data returned')

        vector = product_vector['data']['Get'][self.items_class][0]['_additional']['vector']

        query_builder = self.client.query.get(
            self.items_class, self.properties
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
            
        try:
            results = query_builder.do()
            if results is None:
                return []
            
            similar_items = results.get('data', {}).get('Get', {}).get(self.items_class, [])
            return similar_items
        except weaviate.exceptions.WeaviateTimeoutError as e:
            raise ServiceUnavailableError("Timeout occurred while retrieving similar items: " + str(e))
        except Exception as e:
            raise ProcessingError(f"Failed to retrieve similar items: {str(e)}")

    def get_all_brands(self):
        try:
            group_by_filter_criteria = {
                "operator": "Equal",
                "path": ["country"],
                "valueString": "es"
            }

            response = (
                self.client.query
                .aggregate(self.items_class)
                .with_group_by_filter(group_by_filter_criteria) 
                .with_meta_count()
                .with_fields("meta { count } groupedBy { path value }")
                .do()
            )
            if response is None:
                return []
            return [item['value'] for item in response.get('data', {}).get('Aggregate', {}).get(self.items_class, {}).get('groupedBy', [])]
        except weaviate.exceptions.WeaviateTimeoutError as e:
            raise ProcessingError("Timeout occurred while retrieving brands: " + str(e))
        except Exception as e:
            raise ProcessingError(f"Failed to retrieve brands: {str(e)}")