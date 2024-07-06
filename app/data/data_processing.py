from app.data.data_access import DataAccess
from app.data.queries import combined_tags_query, combined_query
from app.common.utilities import create_item_data
import torch
import pandas as pd
import numpy as np
import faiss

class ImageDataProcessor:
    def __init__(self, data_access):
        self.data_access = data_access
        self.images_tags_df = self.load_images_tags_df()
        self.combined_df = self.load_combined_df()
        self.filtered_df = self.filter_entries_without_features()
        self.e_img_clip_dict = self.extract_e_img_clip()
        self.all_images_df = self.extract_all_images_df()
        self.images_df = self.extract_all_images_df_with_features()

    def load_images_tags_df(self):
        df = self.data_access.query_datasets_to_df(combined_tags_query)
        df = df[['img_url', 'value']].dropna().drop_duplicates().set_index('img_url')
        return df

    def load_combined_df(self):
        df = self.data_access.query_datasets_to_df(combined_query)
        return df.drop_duplicates(subset='shop_link').reset_index(drop=True)

    def extract_all_images_df(self):
        df = self.combined_df.drop(columns=['features'])
        df['old_price'] = df['old_price'].fillna(df['price'])
        return df

    def extract_all_images_df_with_features(self):
        df = self.all_images_df[
            self.all_images_df['img_url'].isin(self.e_img_clip_dict.keys())
            ].set_index('img_url')
        return df

    def filter_entries_without_features(self):
        return self.combined_df[~self.combined_df['features'].isnull()]

    def extract_e_img_clip(self):
        return {row['img_url']: torch.tensor(row['features']).reshape(1, -1).float() for _, row in self.filtered_df.iterrows()}

class ImageIndexBuilder:
    def __init__(self, e_img_clip_dict, images_df, e_img_clip_df):
        self.e_img_clip_dict = e_img_clip_dict
        self.images_df = images_df
        self.e_img_clip_df = e_img_clip_df
        self.index_faiss = self.build_index()

    def build_index(self):
        embeddings = np.array([embedding.numpy().flatten() for embedding in self.e_img_clip_dict.values()], dtype=np.float32)
        index_faiss = faiss.IndexFlatL2(embeddings.shape[1])
        index_faiss.add(embeddings)
        return index_faiss

    def faiss_search(self, k, input_shop_link, input_embedding):
        _, indices = self.index_faiss.search(np.array([input_embedding], dtype=np.float32), k + 1)

        similar_images = []
        shop_link_similarities = {}
        for i, idx in enumerate(indices[0][1:]):  # Skip the first index if it's the query itself
            img_url = self.e_img_clip_df.index[idx]
            row = self.images_df.loc[img_url]
            row['img_url'] = img_url
            shop_link = row['shop_link']
            if shop_link == input_shop_link:
                continue
            similarity_score = 1 - i / k

            if shop_link in shop_link_similarities:
                if similarity_score > shop_link_similarities[shop_link]:
                    shop_link_similarities[shop_link] = similarity_score
                    item_data = create_item_data(row)
                    item_data['similarity'] = similarity_score
                    similar_images.append(item_data)
            else:
                item_data = create_item_data(row)
                item_data['similarity'] = similarity_score
                similar_images.append(item_data)
        return similar_images
