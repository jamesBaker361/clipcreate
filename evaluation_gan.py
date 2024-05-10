import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir

import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from torchmetrics.image.inception import InceptionScore
from generator_src import Generator
from huggingface_hub import hf_hub_download, ModelCard, upload_file
from sentence_transformers import SentenceTransformer
from torchvision.transforms import ToPILImage
from datasets import Dataset,load_dataset
import numpy as np
from accelerate import Accelerator
import wandb

from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
import random
import argparse
from static_globals import *

random.seed(1234)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description="evaluation")
parser.add_argument(
    "--model_list",
    type=str,
    nargs="*",
    help="Path to pretrained models or model identifiers from huggingface.co/models.",
)

parser.add_argument(
    "--conditional_model_list",
    type=str,
    nargs="*",
    help="Path to pretrained models or model identifiers from huggingface.co/models where conditional==True",
)

parser.add_argument("--dataset_name",type=str,help="hf dataset for test prompts",default="jlbaker361/wikiart-balanced1000")

parser.add_argument("--hf_dir",type=str,default="jlbaker361/evaluation-gan",help="hf dir to push to")

parser.add_argument("--image_root_dir",type=str,default="/scratch/jlb638/gan_evaluation_images/")
parser.add_argument("--limit",type=int,default=150,  help="how many samples to make")

parser.add_argument("--gen_z_dim",type=int,default=100,help="dim latent noise for generator")
parser.add_argument("--image_dim", type=int,default=512)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/gan-eval/")

def evaluate(args):
    aesthetic_fn=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)
    if args.conditional_model_list is None:
        args.conditional_model_list=[]
    if args.model_list is None:
        args.model_list=[]
    model_dict={
        model: Generator(args.gen_z_dim, args.image_dim,False) for model in args.model_list
    } | {
        model: Generator(args.gen_z_dim, args.image_dim,True) for model in args.conditional_model_list
    }
    result_dict={}
    src_dict={
        "image":[],
        "model":[],
        "score":[]
    }
    hf_dataset=load_dataset(args.dataset_name,split="test")
    inception = InceptionScore(normalize=True)
    prompt_list=[[t,n] for t,n in zip(hf_dataset["text"], hf_dataset["name"])]
    random.shuffle(prompt_list)
    prompt_list=prompt_list[:args.limit]
    prompt_list=["null" for _ in range(args.limit)]
    sentence_encoder=SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
    for model,gen in model_dict.items():
        model_name=model[model.rfind("/")+1:]
        os.makedirs(f"{args.image_dir}{model_name}",exist_ok=True)
        weights_location=hf_hub_download(model, filename="gen-weights.pickle")
        if torch.cuda.is_available():
            state_dict=torch.load(weights_location)
        else:
            state_dict=torch.load(weights_location, map_location=torch.device("cpu"))
        gen.load_state_dict(state_dict)
        result_dict[model]={}
        total_score=0.0
        score_list=[]
        image_list=[]
        for i,prompt in enumerate(prompt_list):
            noise=torch.randn(1,args.gen_z_dim, 1, 1)
            text_encoding=torch.tensor(sentence_encoder.encode(prompt))
            image=gen(noise,text_encoding )
            image_list.append(image)
            pil_image=ToPILImage()( image[0])
            pil_image.save(f"{args.image_dir}{model_name}/{i}.png")
            src_dict["image"].append(pil_image)
            src_dict["model"].append(model)
            score,_=aesthetic_fn(image.detach(),{},{})
            score=score.detach().numpy()[0]
            src_dict["score"].append(score)
            score_list.append(score)
            total_score+=score
        score_std=np.std(score_list)
        result_dict[model]["std"]=score_std
        result_dict[model]["mean"]=total_score/len(score_list)
        print(f"total score {model} mean {total_score/len(score_list)} {total_score} std {score_std}")
        image_tensor=torch.cat(image_list)
        inception.update(image_tensor)
        inception_mean, inception_std=inception.compute()
        print("inception mean", inception_mean, "inceptstion std", inception_std)
        result_dict[model]["inception_mean"]=float(inception_mean)
        result_dict[model]["inception_src"]=float(inception_std)
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