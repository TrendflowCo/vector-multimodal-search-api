import torch
import clip
from joblib import Memory
from app.models.clip_models import compute_text_embeddings
from app.config.config import CACHE_DIR
from app.common.utilities import get_img
from app.data.templates import templates, templates_with_adjectives, garment_types

# Initialize memory cache for storing embeddings
memory = Memory(CACHE_DIR, verbose=0)

class TextService:
    def __init__(self):
        self.templates = templates
        self.templates_with_adjectives = templates_with_adjectives
        self.garment_types = garment_types

    def get_text_embeddings(self, image_url):
        """
        Retrieve or compute image embeddings for a given image URL.
        """
        return compute_text_embeddings(image_url)
        
    def generate_texts(self, prompt):
        """
        Generate texts from prompt and templates.
        
        Args:
            prompt (str): Prompt text.
        
        Returns:
            texts (list): List of generated texts.
        """
        if any([x in self.garment_types for x in prompt.split(' ')]):
            return [template.format(prompt) for template in self.templates]
        else:
            return [template.format(prompt) for template in self.templates_with_adjectives]

    def clean_prompt_text(self, prompt):
        """
        Clean prompt text.
        
        Args:
            prompt (str): Prompt text.
        
        Returns:
            prompt (str): Cleaned prompt text.
        """
        if prompt.startswith('a '):
            prompt = prompt[2:]
        if prompt.startswith('an '):
            prompt = prompt[3:]
        if prompt.startswith('the '):
            prompt = prompt[4:]
        prompt = prompt.lower().replace('style', '')
        prompt = prompt.replace('fashion', '')
        prompt = prompt.replace('inspiration', '')
        prompt = prompt.lower()

        return prompt
