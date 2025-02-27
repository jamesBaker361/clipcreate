import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import matplotlib.pyplot as plt
 
from static_globals import WIKIART_STYLES
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import argparse

from datasets import Dataset,load_dataset
# Generate random pixel data
random_pixels = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

# Create image from array
image = Image.fromarray(random_pixels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Distortion = 1/n * Σ(distance(point, centroid)^2)

def calculate_distortion(point_list,center_list,label_list):
    total=0.0
    for point,label in zip(point_list,label_list):
        center=center_list[label]
        total+=np.linalg.norm(center-point)
    return total/len(point_list)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",return_tensors="pt",do_rescale=False)
#processor.to(device)
model=CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model.to(device)

def get_text_embeddings(text_list):
    inputs=processor(text=text_list,images=image,return_tensors="pt",padding=True)
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(device)
    return model(**inputs).text_embeds.detach().numpy()

def get_image_embeddings(image_list):
    inputs=processor(text=["wow"],images=image_list,return_tensors="pt",padding=True)
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(device)
    return model(**inputs).image_embeds.cpu().detach().numpy()

def compare_distortion(embeds,n_cluster_list,root_path):
    result_dict={}
    for n in n_cluster_list:
        k_means=KMeans(n_clusters=n, random_state=0, n_init="auto").fit(embeds)
        center_list=k_means.cluster_centers_
        np.save(root_path.format(n),center_list)
        label_list=k_means.labels_
        distortion=calculate_distortion(embeds,center_list,label_list)
        result_dict[n]=distortion
    return result_dict

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=str,default="text")
    parser.add_argument("--dataset",type=str, default="jlbaker361/wikiart-balanced500")
    parser.add_argument("--base_path",type=str,default="centers/")
    
    args=parser.parse_args()

    print(args)
    os.makedirs(args.base_path,exist_ok=True)
    wikiart_dataset=load_dataset(args.dataset,split="train")
    image_list=[row["image"] for row in wikiart_dataset]
    print(f"n images = {len(image_list)}")
    if args.data=="text":
        embeds=get_text_embeddings(WIKIART_STYLES)
        path=args.base_path+"_{}"
        fig_path="text_dist.png"
    else:
        embeds=[get_image_embeddings([image])[0] for image in image_list]
        dataset=args.dataset.split("/")[-1]
        path=args.base_path+dataset+"_{}"
        fig_path=dataset+"_dist.png"
    result_dict=compare_distortion(embeds,[x for x in range(2,15)],path)
    x=[k for k in result_dict.keys()]
    y=[v for v in result_dict.values()]
    plt.scatter(x, y)
    plt.xlabel('n clusters')
    plt.ylabel('distortion')
    plt.savefig(fig_path)
    print(fig_path)
    print("all done :)))")