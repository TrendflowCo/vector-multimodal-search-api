import torch
import torch.nn.functional as F
from app.services.text_service import TextService
from app.services.image_service import ImageService
from app.models.clip_models import compute_text_embeddings, compute_image_embeddings
from app.common.utilities import extract_negative_keywords, concatenate_embeddings, create_item_data
from app.config.config import MAX_K
from app.common.utilities import translate
import numpy as np
import pandas as pd


sale_wording = ['on sale', 'sales', 'sale', 'clearance']
conn_brand_words = ['by', 'from', 'of']
conn_cat_words = ['for']
conn_price_words = ['under', 'below', 'for less than', 'less than', 'less']
curr_symbols = ['$', '€', 'dollars', 'euros', 'dollar', 'euro', 'usd', 'eur', 'us$']

class SimilarityService:
    def __init__(self, images_df, e_img_clip_dict, e_img_clip_cat, e_img_clip_df, image_data_dict, image_index_builder):
        self.text_service = TextService()
        self.image_service = ImageService()
        self.images_df = images_df
        self.e_img_clip_dict = e_img_clip_dict
        self.e_img_clip_cat = e_img_clip_cat
        self.e_img_clip_df = e_img_clip_df
        self.image_data_dict = image_data_dict
        self.all_brands = images_df['brand'].str.lower().unique().tolist()
        self.all_categories = images_df['category'].str.lower().unique().tolist()
        self.image_index_builder = image_index_builder

    def compute_similarity_between_text_and_image(self, text, image_url):
        text_features = self.text_service.compute_text_embeddings(text)
        image_features = self.image_service.get_image_embeddings(image_url)
        
        return self.compute_cosine_similarity(text_features, image_features)


    def compute_similarity_between_images(self, image_url1, image_url2):
        image_features1 = self.image_service.get_image_embeddings(image_url1)
        image_features2 = self.image_service.get_image_embeddings(image_url2)
        
        return self.compute_cosine_similarity(image_features1, image_features2)

    def compute_cosine_similarity(self, features1, features2):
        return features1 @ features2.T
        
            
    def compute_cosine_similarity_normalized(self, features1, features2):
        features1_normalized = F.normalize(features1, dim=1)
        features2_normalized = F.normalize(features2, dim=1)
        return self.compute_cosine_similarity(features1_normalized, features2_normalized)

    def retrieve_most_similar_items(self, input_id, k):
        """
        Retrieve most similar items to a given product.
        
        Args:
            input_id (str): ID of the product.
            k (int): Number of similar items to retrieve.
        
        Returns:
            similar_images (list): List of similar items.
        """
        input_index = self.images_df[self.images_df['id'] == input_id].index
        if len(input_index)==0:
            return None

        input_img_url = input_index[0]
        input_shop_link =  self.images_df.loc[input_img_url]['shop_link']
        input_embedding = self.e_img_clip_dict[input_img_url].numpy().flatten()
        similar_images = self.image_index_builder.faiss_search(k, input_shop_link, input_embedding)
        
        return similar_images

    def compute_image_image_similarities(self, image_url, max_k, threshold):
        """
        Compute similarities between images.
        
        Args:
            self.e_img_clip_cat (torch.tensor): Image embeddings.
            image_url (str): Image URL.
            max_k (int): Maximum number of similar items to retrieve.
            threshold (float): Minimum similarity score.
        
        
        Returns:
            similar_items (list): List of similar items.
        """
        e_query_img = image_service.get_image_embeddings(image_url)
        similarity = self.compute_cosine_similarity_normalized(self.e_img_clip_cat, e_query_img)
        if max_k is not None:
            top_idx = np.array(list(set(similarity.topk(max_k).indices.ravel().tolist())))
        else:
            top_idx = np.arange(similarity.shape[1])  # All indices
        print('Threshold', threshold)
        top_idx = top_idx[(similarity[:, top_idx].max(0).values > threshold).numpy()]

        similar_items = []

        for idx in top_idx:
            idx = idx.item()
            idx_ = self.e_img_clip_df.index[idx]
            item_data = self.image_data_dict[idx_]
            similarity_max = similarity[:, idx].max().item()
            item_data['similarity'] = similarity_max
            similar_items.append(item_data)
        return similar_items

    def get_image_query_similarity_score(self, query, img_url):
        """
        Get similarity score between an image query and an image URL.
        
        Args:
            query (str): Image query.
            img_url (str): Image URL.
        
        Returns:
            similarity_score (float): Similarity score.
            error (str): Error message.
        """
        try:
            query = self.clean_prompt_text(query)
            texts = self.generate_texts(query)
            e_text_cat = torch.cat([self.compute_text_embeddings(t) for t in texts]).to('cpu')
            similarity_score = self.compute_cosine_similarity(e_img_clip_dict[img_url], e_text_cat).max().item()
            return similarity_score, None
        except Exception as e:
            return None, str(e)


        
    def compute_text_image_similarities(self, query, max_k, threshold):
        """
        Compute similarities between text and images.
        
        Args:
            self.e_img_clip_cat (torch.tensor): Image embeddings.
            query (str): Text query.
            max_k (int): Maximum number of similar items to retrieve.
            threshold (float): Minimum similarity score.
        
        
        Returns:
            similar_items (list): List of similar items.
        """
        query = self.text_service.clean_prompt_text(query)
        texts = self.text_service.generate_texts(query)
        text_embeddings_list = [self.text_service.get_text_embeddings(t) for t in texts]
        e_text_cat = concatenate_embeddings(text_embeddings_list)
        similarity = self.compute_cosine_similarity_normalized(e_text_cat, self.e_img_clip_cat)
        if max_k is not None:
            top_idx = np.array(list(set(similarity.topk(max_k).indices.ravel().tolist())))
        else:
            top_idx = np.arange(similarity.shape[1])  # All indices
        print('Threshold', threshold)
        top_idx = top_idx[(similarity[:, top_idx].max(0).values > threshold).numpy()]

        similar_items = []
        
        for idx in top_idx:
            idx = idx.item()
            idx_ = self.e_img_clip_df.index[idx]
            item_data = self.image_data_dict[idx_]
            similarity_max = similarity[:, idx].max().item()
            item_data['similarity'] = similarity_max
            similar_items.append(item_data)
            
        return similar_items

    def retrieve_filtered_images(self, filters):
        """
        Retrieve filtered images.
        
        Args:
            self.images_df (pd.DataFrame): Dataframe of images.
            filters (dict): Filters.
        
        Returns:
            filtered_items (pd.DataFrame): Filtered images.
        """
        filtered_items = self.images_df.copy()

        if filters.get('query'):
            
            if filters['language'] != 'en':
                print(f"Translation from {filters['language']} to en: \n")
                print(f"Original: {filters['query']}\n")
                filters['query'] = translate(filters['query'], filters['language'])
                print(f"English: {filters['query']}")
            threshold = filters.get('threshold', 0.21)
            
            if filters['image_url']:
                similar_items = self.compute_image_image_similarities(img_url, max_k=MAX_K, threshold=threshold)
            else:
                filters['query'] = filters['query'].lower()
                
                if any([w in filters['query'] for w in sale_wording]):
                    filters['on_sale'] = True
                    for w in sale_wording:
                        filters['query'] = filters['query'].replace(w, '')
                query_brands = [brand for brand in self.all_brands if brand in filters['query']]
                
                if len(query_brands)>0:
                    filters['brands'] = ','.join(query_brands)
                    for brand in query_brands:
                        filters['query'] = filters['query'].replace(brand+"'s", '').replace(brand, '')
                    for w in conn_brand_words:
                        filters['query'] = filters['query'].replace(w, '')

                query_categories = []
                for w in filters['query'].split(' '):
                    w_ = w.replace('s', '').replace("'", '')
                    if w_ in self.all_categories:
                        query_categories.append(w_)
                
                if len(query_categories)>0:
                    filters['category'] = query_categories[0]
                    for cat in query_categories:
                        filters['query'] = filters['query'].replace(cat+"'s", '').replace(cat, '')
                    for w in conn_cat_words:
                        filters['query'] = filters['query'].replace(w, '')
                
                if any([w in filters['query'] for w in curr_symbols]):
                    price_pattern = r'(\d+(?:\.\d{1,2})?)\s*(?:dollars|dollar|euros|euro|\$|€)'
                    price_match = re.search(price_pattern, filters['query'], re.IGNORECASE)
                    if price_match:
                        # Extract the matched price as a float
                        filters['max_price'] = float(price_match.group(1))
                        for curr in curr_symbols:
                            filters['query'] = filters['query'].replace(curr, '')
                        for w in conn_price_words:
                            filters['query'] = filters['query'].replace(w, '')
                

                negative_keywords = extract_negative_keywords(filters['query'])
                clean_query = ' '.join([w for w in filters['query'].split(' ') if w.replace('-', '') not in negative_keywords])
                similar_items = self.compute_text_image_similarities(clean_query, max_k=MAX_K, threshold=threshold)
        
            if len(similar_items) == 0:
                return None
            
            similar_items_df = pd.DataFrame(similar_items).set_index('img_url')

            if len(negative_keywords)>0:
                blacklist = []
                for keyword in negative_keywords:
                    negative_results = self.compute_text_image_similarities(f'{clean_query} {keyword}', max_k=MAX_K, threshold=threshold)
                    blacklist += [item['img_url'] for item in negative_results]
                similar_items_df = similar_items_df[~similar_items_df.index.isin(blacklist)]

            similar_items_df = similar_items_df.sort_values(by='similarity', ascending=False)
            filtered_items = filtered_items.loc[similar_items_df.index]
            filtered_items['similarity'] = similar_items_df['similarity']
        
        if filters.get('category'):
            filtered_items = filtered_items[filtered_items['category'] == filters['category']]
        
        if filters.get('min_price'):
            filtered_items = filtered_items[filtered_items['price'] >= filters['min_price']]
        
        if filters.get('max_price'):
            filtered_items = filtered_items[filtered_items['price'] <= filters['max_price']]
        
        if filters.get('tags'):
            list_tags = [x.lower() for x in filters[F'tags'].replace("'", "").split(',')]
            images_tags_filtered_df = images_tags_df[images_tags_df['value'].str.lower().isin(list_tags)]
            list_tags_img_ids = images_tags_filtered_df['img_id'].unique().tolist()
            filtered_items = filtered_items[filtered_items['img_id'].isin(list_tags_img_ids)]
        
        if filters.get('brands'):
            list_brands = [x.lower() for x in filters['brands'].replace("'", "").split(',')]
            filtered_items = filtered_items[filtered_items['brand'].str.lower().isin(list_brands)]
        
        if filters.get('on_sale'):
            filtered_items = filtered_items[filtered_items['sale'] == filters['on_sale']]
        
        if filters.get('ids'):
            list_ids = filters['ids'].replace("'", "").split(',')
            filtered_items = filtered_items[filtered_items['id'].isin(list_ids)]
        
        return filtered_items.reset_index().drop_duplicates(['img_url', 'shop_link'])

    def retrieve_similar_images(self, image_url, top_k=10):
        """
        Retrieve similar images given an image URL.
        """
        query_embedding = self.image_service.get_image_embeddings(image_url)
        input_shop_link = self.images_df.loc[image_url]['shop_link']
        similar_images = self.image_index_builder.faiss_search(top_k, input_shop_link, query_embedding)
        return similar_images
