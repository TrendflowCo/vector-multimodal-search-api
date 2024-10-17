import unittest
import json
from app import create_app
from app.services.weviate_service import WeaviateService
from unittest.mock import patch, MagicMock

class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.weaviate_service = WeaviateService()

    def test_search_endpoint(self):
        with patch.object(WeaviateService, 'search_with_text') as mock_search:
            mock_search.return_value = (
                [{'id': '1', 'name': 'Test Product', 'price': 100}],
                1
            )
            response = self.client.get('/search?query=test&page=1&limit=10')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('results', data)
            self.assertEqual(len(data['results']), 1)
            self.assertEqual(data['total_results'], 1)

    def test_product_endpoint(self):
        with patch.object(WeaviateService, 'get_product_details') as mock_get_product:
            mock_get_product.return_value = {
                'id': '1',
                'name': 'Test Product',
                'price': 100,
                'tags': ['tag1', 'tag2']
            }
            response = self.client.get('/product?id=1')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('result', data)
            self.assertEqual(data['result']['name'], 'Test Product')

    def test_most_similar_items_endpoint(self):
        with patch.object(WeaviateService, 'get_similar_items') as mock_get_similar:
            mock_get_similar.return_value = [
                {'id': '2', 'name': 'Similar Product', 'price': 90}
            ]
            response = self.client.get('/most_similar_items?id=1&top_k=5')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('results', data)
            self.assertEqual(len(data['results']), 1)

    def test_brands_list_endpoint(self):
        with patch.object(WeaviateService, 'get_all_brands') as mock_get_brands:
            mock_get_brands.return_value = ['Brand1', 'Brand2', 'Brand3']
            response = self.client.get('/brands_list')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('brand_list', data)
            self.assertEqual(len(data['brand_list']), 3)

    def test_ingest_endpoint(self):
        with patch('app.services.ingest_service.IngestService.ingest_data') as mock_ingest:
            mock_ingest.return_value = 100
            response = self.client.post('/ingest')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('message', data)
            self.assertIn('ingested_count', data)
            self.assertEqual(data['ingested_count'], 100)

    def test_search_endpoint_error(self):
        with patch.object(WeaviateService, 'search_with_text') as mock_search:
            mock_search.side_effect = Exception("Test error")
            response = self.client.get('/search?query=test')
            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)
            self.assertIn('error', data)

    def test_product_endpoint_not_found(self):
        with patch.object(WeaviateService, 'get_product_details') as mock_get_product:
            mock_get_product.return_value = None
            response = self.client.get('/product?id=999')
            self.assertEqual(response.status_code, 404)
            data = json.loads(response.data)
            self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
