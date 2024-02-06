import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir

import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from torchmetrics.image.inception import InceptionScore
from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
from datasets import Dataset,load_dataset

import argparse
parser = argparse.ArgumentParser(description="download and evaluate a dataset")
parser.add_argument("--dataset_name",type=str, help="dataset to download from hf")

def evaluate(args):
    hf_dataset=load_dataset(args.dataset_name,split="train")
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

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    evaluate(args)