from accelerate import Accelerator
from experiment_helpers.measuring import get_metric_dict
from experiment_helpers.static_globals import AESTHETIC_SCORE,IMAGE_REWARD

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
import os
from PIL import Image
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

parser.add_argument("--model",type=str,default="jlbaker361/dcgan-vanilla")

parser.add_argument("--dataset_name",type=str,help="hf dataset for test prompts",default="jlbaker361/wikiart-balanced1000")

parser.add_argument("--hf_dir",type=str,default="jlbaker361/evaluation-gan",help="hf dir to push to")

parser.add_argument("--image_root_dir",type=str,default="/scratch/jlb638/gan_evaluation_images/")
parser.add_argument("--limit",type=int,default=150,  help="how many samples to make")

parser.add_argument("--gen_z_dim",type=int,default=100,help="dim latent noise for generator")
parser.add_argument("--image_dim", type=int,default=512)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/gan-eval/")
parser.add_argument("--project_name",type=str,default="gan-evaluation")

def evaluate(args):
    prompt_list=["art" for _ in range(args.limit)]
    sentence_encoder=SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
    accelerator=Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name=args.project_name, config=vars(args))
    model_name=args.model[args.model.rfind("/")+1:]
    os.makedirs(f"{args.image_dir}/{model_name}",exist_ok=True)
    weights_location=hf_hub_download(args.model, filename="gen-weights.pickle")
    if torch.cuda.is_available():
        state_dict=torch.load(weights_location)
    else:
        state_dict=torch.load(weights_location, map_location=torch.device("cpu"))
    gen=Generator(args.gen_z_dim, args.image_dim,False)
    gen.load_state_dict(state_dict)
    gen.eval()
    gen.requires_grad_(False)
    gen.to(accelerator.device)
    gen=accelerator.prepare(gen)
    image_list=[]
    for i,prompt in enumerate(prompt_list):
        noise=torch.randn(1,args.gen_z_dim, 1, 1)
        text_encoding=torch.tensor(sentence_encoder.encode(prompt))
        image=gen(noise,text_encoding )[0]
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        image_list.append(image)
        path=f"{args.image_dir}/{model_name}/{i}.png"
        image.save(path)
        accelerator.log({
            f"{model_name}_{i}":wandb.Image(path)
        })
    metric_dict=get_metric_dict(prompt_list, image_list, [image],accelerator)
    for metric in [AESTHETIC_SCORE, IMAGE_REWARD]:
        accelerator.log({
            metric:metric_dict[metric]
        })
        print(metric, metric_dict[metric])
    print("all done :)))")

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    evaluate(args)