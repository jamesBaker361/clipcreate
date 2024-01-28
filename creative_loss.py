import os
from transformers import BlipProcessor, BlipForConditionalGeneration,BlipModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from res_net_src import ResNet
from discriminator_src import Discriminator,SquarePad
from torchvision import transforms
from huggingface_hub import hf_hub_download
from torch.nn import Softmax
import torch
import numpy as np
import torch

cache_dir="/scratch/jlb638/trans_cache"

def cross_entropy_per_sample(y_pred, y_true):
    loss = 0.0
    # Doing cross entropy Loss
    for i in range(len(y_pred)):
        loss = loss + (-1 * y_true[i]*torch.log(y_pred[i]))
    return loss

def cross_entropy(pred,true):
    return [cross_entropy_per_sample(y_pred,y_true) for y_pred,y_true in zip(pred,true)]

def cross_entropy_per_sample_dcgan(y_pred, y_true):
    loss = 0.
    # Doing cross entropy Loss
    for i in range(len(y_pred)):
        loss = loss + (-1 * y_true[i]*torch.log(y_pred[i]))
    print(loss)
    return loss

def cross_entropy_dcgan(pred,true):
    return torch.stack([cross_entropy_per_sample(y_pred,y_true) for y_pred,y_true in zip(pred,true)])

def clip_scorer_ddpo(style_list): #https://github.com/huggingface/trl/blob/main/examples/scripts/ddpo.py#L126
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",do_rescale=False)

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        inputs = processor(text=style_list, images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)

        n_classes=len(style_list)
        n_image=images.shape[0]
        uniform=torch.full((n_image, n_classes), fill_value=1.0/n_classes)
        #uniform=torch.normal(0, 5, size=(n_image, n_text))

        #cosine = torch.nn.CosineSimilarity(dim=1) 

        scores =  -1 * cross_entropy(probs,uniform)
        print(scores)
        #scores=torch.normal(0.0, 5.0, size=(1,n_image))
        return scores, {}

    return _fn


def elgammal_dcgan_scorer_ddpo(style_list,image_dim, resize_dim, disc_init_dim,disc_final_dim,repo_id,device=None):
    n_classes=len(style_list)
    model=Discriminator(image_dim,disc_init_dim,disc_final_dim,style_list)
    weights_location=hf_hub_download(repo_id, filename="disc-weights.pickle")
    if torch.cuda.is_available():
        state_dict=torch.load(weights_location)
    else:
        state_dict=torch.load(weights_location, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    
    transform_composition=transforms.Compose([
            SquarePad(),
            transforms.Resize(resize_dim),
            transforms.CenterCrop(image_dim)
    ])

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        images=transform_composition(images)
        _,probs=model(images)
        n_image=images.shape[0]
        print(f"n_image = {n_image}")
        uniform=torch.full((n_image, n_classes), fill_value=1.0/n_classes,device=device)
        print("time to caluclate scores???")
        scores = -1 * cross_entropy_dcgan(probs,uniform)
        print(type(scores))
        return scores, {}
    
    return _fn


def elgammal_resnet_scorer_ddpo(style_list, center_crop_dim):
    n_classes=len(style_list)
    model=ResNet("resnet18", n_classes)
    weights_location=hf_hub_download(repo_id="jlbaker361/resnet-wikiart", filename="resnet-weights.pickle")
    if torch.cuda.is_available():
        state_dict=torch.load(weights_location)
    else:
        state_dict=torch.load(weights_location, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)


    transform_composition=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(center_crop_dim),
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        images=transform_composition(images)
        probs=model(images)
        n_image=images.shape[0]
        uniform=torch.full((n_image, n_classes), fill_value=1.0/n_classes)
        #uniform=torch.normal(0, 5, size=(n_image, n_text))

        #cosine = torch.nn.CosineSimilarity(dim=1) 

        scores = -1 * cross_entropy(probs,uniform)

        return scores, {}
    
    return _fn
