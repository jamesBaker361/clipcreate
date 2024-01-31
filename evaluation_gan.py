import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir

import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from generator_src import Generator
from discriminator_src import Discriminator,GANDataset,UtilDataset
from huggingface_hub import hf_hub_download, ModelCard, upload_file
from torchvision.transforms import ToPILImage
from datasets import Dataset,load_dataset
import numpy as np

from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
import random
import argparse
from static_globals import *

random.seed(1234)

parser = argparse.ArgumentParser(description="evaluation")
parser.add_argument(
    "--model_list",
    type=str,
    nargs="*",
    help="Path to pretrained models or model identifiers from huggingface.co/models.",
)

parser.add_argument("--hf_dir",type=str,default="jlbaker361/evaluation-gan",help="hf dir to push to")

parser.add_argument("--image_root_dir",type=str,default="/scratch/jlb638/gan_evaluation_images/")
parser.add_argument("--limit",type=int,default=150,  help="how many samples to make")

parser.add_argument("--gen_z_dim",type=int,default=100,help="dim latent noise for generator")
parser.add_argument("--image_dim", type=int,default=512)

def evaluate(args):
    aesthetic_fn=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)
    model_dict={
        model: Generator(args.gen_z_dim, args.image_dim) for model in args.model_list
    }
    result_dict={}
    src_dict={
        "image":[],
        "model":[],
        "score":[]
    }
    for model,gen in model_dict.items():
        weights_location=hf_hub_download(model, filename="gen-weights.pickle")
        if torch.cuda.is_available():
            state_dict=torch.load(weights_location)
        else:
            state_dict=torch.load(weights_location, map_location=torch.device("cpu"))
        gen.load_state_dict(state_dict)
        result_dict[model]={}
        total_score=0.0
        score_list=[]
        for i in range(args.limit):
            noise=torch.randn(1,100, 1, 1)
            image=gen(noise)
            src_dict["image"].append(ToPILImage()( image[0]))
            src_dict["model"].append(model)
            score,_=aesthetic_fn(image.detach(),{},{})
            score=score.detach().numpy()[0]
            src_dict["score"].append(score)
            score_list.append(score)
            total_score+=score
        score_std=np.std(score_list)
        result_dict[model]["std"]=score_std
        result_dict[model]["mean"]=total_score/len(score_list)
        print(f"total score {model} {total_score} std {score_std}")
    Dataset.from_dict(src_dict).push_to_hub(args.hf_dir)
    model_card_content=f"created a total of {len(score_list)} images \n"
    for model,metric_dict in result_dict.items():
        model_card_content+="\n"+model
        for m_name,m_value in metric_dict.items():
            model_card_content+=f" {m_name}: {m_value} "
    with open("tmp.md","w+") as file:
        file.write(model_card_content)
    upload_file(path_or_fileobj="tmp.md", 
                path_in_repo="README.md",
                repo_id=args.hf_dir,
                repo_type="dataset")


if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    evaluate(args)