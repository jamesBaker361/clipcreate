import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
import torchvision.transforms as transforms
import torch
from accelerate import Accelerator
import wandb
from static_globals import *
torch.manual_seed(0)
import argparse

parser=argparse.ArgumentParser("negative prompting")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--dir",type=str,default="images")
parser.add_argument("--n_images",type=int,default=10)
parser.add_argument("--prompt",type=str,default=" ")
parser.add_argument("--base_model",type=str, default="stabilityai/stable-diffusion-2-base")

#lora_generator= torch.Generator(device="cpu").manual_seed(0)

from call_neg import call_multi_neg, call_vanilla
def main(args):
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name="negative_prompting",init_kwargs={
            "wandb":
                {"config":{
                    "negative_prompts":WIKIART_STYLES,
                    "n_steps":args.num_inference_steps,
                    "prompt":args.prompt
                }
            }
        })

    pipeline=DefaultDDPOStableDiffusionPipeline(args.base_model)
    #pipeline.sd_pipeline.to(device)
    #pipeline.sd_pipeline=accelerator.prepare(pipeline.sd_pipeline)
    #generator = torch.Generator(device=accelerator.device).manual_seed(0)

    for x in range(args.n_images):
        neg_img=call_multi_neg(pipeline.sd_pipeline,"painting",num_inference_steps=args.num_inference_steps,
                        negative_prompt=WIKIART_STYLES).images[0]
        path=f"{args.dir}/img{x}.png"
        neg_img.save(path)
        accelerator.log({"neg_image": wandb.Image(path)},step=x)
        vanilla_img=pipeline("painting",num_inference_steps=args.num_inference_steps).images[0]
        path=f"{args.dir}/vanilla_img{x}.png"
        vanilla_img.save(path)
        accelerator.log({"vanilla_image": wandb.Image(path)},step=x)

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    main(args)