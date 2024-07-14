import unittest
import requests

BASE_URL = "http://localhost:5000/api"

class TestAPIEndpoints(unittest.TestCase):

    def test_product_endpoint(self):
        params = {
            'id': 'some_product_id',
            'language': 'en',
            'country': 'US',
            'currency': 'USD'
        }
        response = requests.get(f"{BASE_URL}/product", params=params)
        self.assertEqual(response.status_code, 200)
        self.assertIn('result', response.json())

    def test_search_endpoint(self):
        params = {
            'query': 'some_query',
            'page': 1,
            'limit': 10,
            'sortBy': 'price',
            'ascending': 'true',
            'threshold': 0.8,
            'maxPrice': 100,
            'minPrice': 10,
            'category': 'electronics',
            'onSale': 'true',
            'tags': 'tag1,tag2',
            'brands': 'brand1,brand2',
            'ids': 'id1,id2',
            'language': 'en',
            'country': 'US',
            'currency': 'USD'
        }
        response = requests.get(f"{BASE_URL}/search", params=params)
        self.assertEqual(response.status_code, 200)
        self.assertIn('results', response.json())

    def test_brands_list_endpoint(self):
        response = requests.get(f"{BASE_URL}/brands_list")
        self.assertEqual(response.status_code, 200)
        self.assertIn('brand_list', response.json())

    def test_most_similar_items_endpoint(self):
        params = {
            'id': 'some_product_id',
            'top_k': 10,
            'country': 'US',
            'currency': 'USD'
        }
        response = requests.get(f"{BASE_URL}/most_similar_items", params=params)
        self.assertEqual(response.status_code, 200)
        self.assertIn('results', response.json())

    def test_image_query_similarity_endpoint(self):
        params = {
            'query': 'some_image_query',
            'img_url': 'https://static.zara.net/assets/public/075e/e927/a9954b388618/519b7a116a15/00518063802-p/00518063802-p.jpg?ts=1713514518983&w=824'
        }
        response = requests.get(f"{BASE_URL}/image_query_similarity", params=params)
        self.assertEqual(response.status_code, 200)
        self.assertIn('similarity_score', response.json())

if __name__ == "__main__":
    unittest.main()
