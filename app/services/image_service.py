import torch
import numpy as np
from app.models.clip_models import compute_image_embeddings
from app.common.utilities import get_img, create_item_data
from app.config import CACHE_DIR
from joblib import Memory

# Initialize memory cache for storing embeddings
memory = Memory(CACHE_DIR, verbose=0)

class ImageService:
    def __init__(self):
        pass


    @memory.cache
    def get_image_embeddings(self, image_url):
        """
        Retrieve or compute image embeddings for a given image URL.
        """
        return compute_image_embeddings(image_url)

    def get_image_by_url(self, image_url):
        """
        Fetch an image from a URL and return as a PIL Image object.
        """
        return get_img(image_url)


    def add_image_to_index(self, image_url):
        """
        Add a new image to the FAISS index after computing its embeddings.
        """
        embedding = self.get_image_embeddings(image_url).numpy().flatten()
        self.index_faiss.add(np.array([embedding], dtype=np.float32))
        # Update the e_img_clip dictionary
        self.e_img_clip[image_url] = torch.from_numpy(embedding)

    def remove_image_from_index(self, image_url):
        """
        Remove an image from the FAISS index. This operation is not straightforward in FAISS
        and typically requires rebuilding the index.
        """
        # This is a placeholder to indicate the need for such functionality.
        # Actual implementation will depend on specific requirements and FAISS capabilities.
        pass
