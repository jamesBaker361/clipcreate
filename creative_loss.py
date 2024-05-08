from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from discriminator_src import Discriminator,SquarePad
from torchvision.transforms import PILToTensor
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import torch
from scipy.special import softmax
import random
import ImageReward as reward
import string
import PIL
from typing import List
def generate_random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))
reward_cache="/scratch/jlb638/reward_symbolic/"+generate_random_string(10)

from PIL import Image

cache_dir="/scratch/jlb638/trans_cache"

def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images

def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    if np.max(images)<=1 and np.min(images)<0: #between -1,1
        images=(images*0.5)+0.5
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def cross_entropy_per_sample(y_pred, y_true):
    loss = 0.0
    # Doing cross entropy Loss
    for i in range(len(y_pred)):
        loss = loss + (-1 * y_true[i]*np.log(y_pred[i]))
    return loss

def mse(y_true,y_pred):
    assert len(y_true)==len(y_pred)
    loss=0.
    for i in range(len(y_pred)):
        loss = loss + (y_true[i]-y_pred[i])**2
    return loss

def cross_entropy(pred,true):
    return [cross_entropy_per_sample(y_pred,y_true) for y_pred,y_true in zip(pred,true)]

def cross_entropy_per_sample_dcgan(y_pred, y_true):
    loss = 0.
    # Doing cross entropy Loss
    for i in range(len(y_pred)):
        loss = loss + (-1 * y_true[i]*torch.log(y_pred[i]))
    return loss

def cross_entropy_dcgan(pred,true):
    return torch.stack([cross_entropy_per_sample(y_pred,y_true) for y_pred,y_true in zip(pred,true)])

classification_loss=torch.nn.CrossEntropyLoss()

def clip_scorer_ddpo(style_list): #https://github.com/huggingface/trl/blob/main/examples/scripts/ddpo.py#L126
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        if type(images)==torch.Tensor or type(images)==torch.FloatTensor:
            images=pt_to_numpy(images)
            images=numpy_to_pil(images)
        elif type(images)==list:
            if type(images[0])==torch.Tensor or type(images[0])==torch.FloatTensor:
                images=[
                    numpy_to_pil(pt_to_numpy(image)) for image in images
                ]
            elif type(images[0])!=Image:
                print(f"image of type {type(images[0])}")
        else:
            print("type(images)",type(images))

        inputs = processor(images=images,text=style_list, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        #probs = logits_per_image.softmax(dim=1)

        n_classes=len(style_list)
        n_image=len(images)

        scores=[]
        for x in range(n_image):
            y_true=[1.0/n_classes] * n_classes
            y_pred=logits_per_image[x]
            scores.append(-1.0* classification_loss(torch.tensor(y_pred),torch.tensor(y_true)))
        
        return scores, {}

    return _fn


def elgammal_dcgan_scorer_ddpo(style_list,image_dim, resize_dim, disc_init_dim,disc_final_dim,repo_id,device=None):
    n_classes=len(style_list)
    model=Discriminator(image_dim,disc_init_dim,disc_final_dim,style_list,False,False)
    weights_location=hf_hub_download(repo_id, filename="disc-weights.pickle")
    if torch.cuda.is_available():
        state_dict=torch.load(weights_location)
    else:
        state_dict=torch.load(weights_location, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict,strict=False)
    if device is not None:
        model.to(device)
    
    def transform_composition(images):
        pil_to_tensor=PILToTensor()
        if type(images)==torch.Tensor or type(images)==torch.FloatTensor:
            if torch.max(images)<=1.0 and torch.min(images)>=0: #between [0,1] -> [-1,1]
                images=(images*2)-1.0
        elif type(images)==list:
            if type(images[0])==torch.Tensor or type(images[0])==torch.FloatTensor:
                if torch.max(images[0])<=1.0 and torch.min(images[0])>=0:
                    images=[
                        (img*2.0)-1.0 for img in images
                    ]
            if type(images[0])==Image:
                images=[
                    pil_to_tensor(img)/128 -1.0 for img in images
                ]



    @torch.no_grad()
    def _fn(images, prompts, metadata):
        images=transform_composition(images)
        images=images.cpu()
        _,probs=model(images,None) #
        n_image=images.shape[0]
        uniform=torch.full((n_image, n_classes), fill_value=1.0/n_classes,device=device)
        scores=[]
        for x in range(n_image):
            y_true=[1.0/n_classes] * n_classes
            y_pred=probs[x]
            scores.append(-1.0 * classification_loss(torch.tensor(y_pred),torch.tensor(y_true)))
        return scores, {}
    
    return _fn

def k_means_scorer(center_list_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    center_list=np.load(center_list_path)
    @torch.no_grad()
    def _fn(images, prompts, metadata):
        if type(images)==torch.Tensor or type(images)==torch.FloatTensor:
            images=pt_to_numpy(images)
            images=numpy_to_pil(images)
        elif type(images)==list:
            if type(images[0])==torch.Tensor or type(images[0])==torch.FloatTensor:
                images=[
                    numpy_to_pil(pt_to_numpy(image)) for image in images
                ]
            elif type(images[0])!=Image:
                print(f"image of type {type(images[0])}")
        else:
            print("type(images)",type(images))

        inputs = processor(images=images,text="text", return_tensors="pt", padding=True)
        outputs = model(**inputs)
        image_embeds=outputs.image_embeds.detach().numpy()

        n_classes=len(center_list)

        scores=[]
        y_true=[1.0/n_classes] * n_classes
        for x in image_embeds:
            y_pred=[]
            for center in center_list:
                try:
                    dist=1.0/ np.linalg.norm(center - x)
                except ZeroDivisionError:
                    dist=1000000
                y_pred.append(dist)
            #y_pred=softmax(y_pred)
            scores.append(-1.0 * classification_loss(torch.tensor(y_pred),torch.tensor(y_true)))
        
        return scores, {}

    return _fn

def image_reward_scorer():
    model=reward.load("/scratch/jlb638/reward-blob",med_config="/scratch/jlb638/ImageReward/med_config.json")

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        if type(images)==torch.Tensor or type(images)==torch.FloatTensor:
            images=pt_to_numpy(images)
            images=numpy_to_pil(images)
        elif type(images)==list:
            if type(images[0])==torch.Tensor or type(images[0])==torch.FloatTensor:
                images=[
                    numpy_to_pil(pt_to_numpy(image)) for image in images
                ]
            elif type(images[0])!=Image:
                print(f"image of type {type(images[0])}")
        else:
            print("type(images)",type(images))

        try:
            return [model.score( prompt,image) for image,prompt in zip(images,prompts)],{}
        except:
            print("failed for [model.score( prompt,image) for image,prompt in zip(images,prompts)],{}")
        try:
            return [model.score( prompt,Image.fromarray(image)) for image,prompt in zip(images,prompts)],{}
        except:
            print("failed for [model.score( prompt,Image.fromarray(image)) for image,prompt in zip(images,prompts)],{}")
        
    return _fn