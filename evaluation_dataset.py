import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir

import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from torchmetrics.image.inception import InceptionScore
from datasets import Dataset,load_dataset
from torchvision.transforms import PILToTensor
import numpy as np

import argparse
parser = argparse.ArgumentParser(description="download and evaluate a dataset")
parser.add_argument("--dataset_name",type=str, help="dataset to download from hf")

def evaluate(args):
    hf_dataset=load_dataset(args.dataset_name,split="train")
    inception = InceptionScore(normalize=True)
    model_image_dict={}
    model_score_dict={}
    for row in hf_dataset:
        ava_score=row["score"]
        model_name=row["model"]
        if model_name not in model_image_dict:
            model_image_dict[model_name]=[]
        if model_name not in model_score_dict:
            model_score_dict[model_name]=[]
        pil_img=row["image"]
        tensor_img=PILToTensor()(pil_img)
        model_image_dict[model_name].append(tensor_img)
        model_score_dict[model_name].append(ava_score)
    for model in model_image_dict.keys():
        image_list=model_image_dict[model]
        image_tensor=torch.stack(image_list)
        inception.update(image_tensor)
        inception_mean, inception_std=inception.compute()
        print(model)
        print("inception mean", inception_mean, "inceptstion std", inception_std)
        score_list=model_score_dict[model]
        print("ava mean", np.mean(score_list), "ava std dev", np.std(score_list))


if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    evaluate(args)