import os
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
from transformers import BlipProcessor, BlipForConditionalGeneration,BlipModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from torch.nn import Softmax
import torch

cache_dir="/scratch/jlb638/trans_cache"

def clip_scorer_ddpo(style_list): #https://github.com/huggingface/trl/blob/main/examples/scripts/ddpo.py#L126
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",cache_dir=cache_dir)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",cache_dir=cache_dir)

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        inputs = processor(text=style_list, images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)

        n_text=len(style_list)
        n_image=images.shape[0]
        uniform=torch.full((n_image, n_text), fill_value=1.0/n_text)
        #uniform=torch.normal(0, 5, size=(n_image, n_text))

        cosine = torch.nn.CosineSimilarity(dim=1) 

        scores = -10*cosine(uniform,probs)
        #scores=torch.normal(0.0, 5.0, size=(1,n_image))
        return scores, {}

    return _fn

