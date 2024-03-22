from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from discriminator_src import Discriminator,SquarePad
from torchvision import transforms
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import torch
from scipy.special import softmax

cache_dir="/scratch/jlb638/trans_cache"

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
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",do_rescale=False)

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        try:
            inputs = processor(images=images,text=style_list, return_tensors="pt", padding=True)
        except ValueError:
            images=images+1
            images=images/2
            inputs = processor(images=images,text=style_list, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        #probs = logits_per_image.softmax(dim=1)

        n_classes=len(style_list)
        n_image=images.shape[0]

        scores=[]
        for x in range(n_image):
            y_true=[1.0/n_classes] * n_classes
            y_pred=logits_per_image[x]
            scores.append(1.0 - classification_loss(y_pred,y_true))
        
        return scores, {}

    return _fn


def elgammal_dcgan_scorer_ddpo(style_list,image_dim, resize_dim, disc_init_dim,disc_final_dim,repo_id,device=None):
    n_classes=len(style_list)
    model=Discriminator(image_dim,disc_init_dim,disc_final_dim,style_list,False)
    weights_location=hf_hub_download(repo_id, filename="disc-weights.pickle")
    if torch.cuda.is_available():
        state_dict=torch.load(weights_location)
    else:
        state_dict=torch.load(weights_location, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict,strict=False)
    if device is not None:
        model.to(device)
    
    transform_composition=transforms.Compose([
            SquarePad(),
            transforms.Resize(resize_dim),
            transforms.CenterCrop(image_dim)
    ])

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        #images=transform_composition(images)
        images=images.cpu()
        _,probs=model(images,None) #
        n_image=images.shape[0]
        uniform=torch.full((n_image, n_classes), fill_value=1.0/n_classes,device=device)
        scores=[]
        for x in range(n_image):
            y_true=[1.0/n_classes] * n_classes
            y_pred=probs[x]
            scores.append(1.0 - classification_loss(y_pred,y_true))
        return scores, {}
    
    return _fn

def k_means_scorer(center_list_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",do_rescale=False)
    center_list=np.load(center_list_path)
    @torch.no_grad()
    def _fn(images, prompts, metadata):
        try:
            inputs = processor(images=images,text="text", return_tensors="pt", padding=True)
        except ValueError:
            images=images+1
            images=images/2
            inputs = processor(images=images,text="text", return_tensors="pt", padding=True)
        outputs = model(**inputs)
        image_embeds=outputs.image_embeds.detach().numpy()

        n_classes=len(center_list)
        n_image=images.shape[0]

        scores=[]
        y_true=[1.0/n_classes] * n_classes
        for x in image_embeds:
            y_pred=[]
            for center in center_list:
                try:
                    dist=1.0/ np.linalg.norm(center - x)
                except ZeroDivisionError:
                    dist=100000
                y_pred.append(dist)
            y_pred=softmax(y_pred)
            scores.append(1.0 - classification_loss(y_pred,y_true))
        
        return scores, {}

    return _fn
