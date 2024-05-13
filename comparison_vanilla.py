import os
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
os.environ["WANDB__SERVICE_WAIT"]="300"
from accelerate import Accelerator
from experiment_helpers.measuring import get_metric_dict
from experiment_helpers.static_globals import AESTHETIC_SCORE,IMAGE_REWARD

from diffusers import StableDiffusionPipeline
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from huggingface_hub import hf_hub_download, ModelCard, upload_file
import wandb
from accelerate import Accelerator
import random
import argparse
from static_globals import *
from ddpo_train_script import get_prompt_fn,load_lora_weights
from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline

parser = argparse.ArgumentParser(description="evaluation vs vanilla pipeline")

parser.add_argument("--model",type=str,default="jlbaker361/dcgan-vanilla")
parser.add_argument("--limit",type=int,default=100,  help="how many samples to make")
parser.add_argument("--project_name",type=str,default="ddpo-vanilla-comparison")
parser.add_argument("--seed",type=int,default=123)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/ddpo-eval-images/")
parser.add_argument("--num_inference_steps",type=int,default=30)

def main(args):
    medium_list=["painting of ","picture of ","drawing of "]
    subject_list=["a man"," a woman"," a landscape"," nature"," a building"," an animal"," shapes", " an object"]
    prompt_list=[]
    for medium in medium_list:
        for subject in subject_list:
            prompt=medium+subject
            prompt_list.append(prompt)
        prompt_list.append(" ")
    accelerator=Accelerator(log_with="wandb")
    accelerator.init_trackers(args.project_name,config=vars(args))
    creative_pipeline=BetterDefaultDDPOStableDiffusionPipeline("stabilityai/stable-diffusion-2-base")
    weight_path=hf_hub_download(repo_id=args.model, filename="pytorch_lora_weights.safetensors",repo_type="model")
    load_lora_weights(creative_pipeline,weight_path)
    pipeline=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
    creative_image_list=[]
    vanilla_image_list=[]
    half_image_list=[]
    third_image_list=[]
    evaluation_prompt_list=[]
    for i in range(args.limit):
        os.makedirs(f"{args.image_dir}{i}",exist_ok=True)
        generator=torch.Generator().manual_seed(i)
        prompt=prompt_list[i %len(prompt_list)]
        evaluation_prompt_list.append(prompt)
        creative_image=creative_pipeline(prompt,num_inference_steps=args.num_inference_steps,generator=generator).images[0]
        creative_path=f"{args.image_dir}{i}/creative.png"
        creative_image.save(creative_path)
        accelerator.log({
            f"{i}/creative":wandb.Image(creative_path)
        })
        creative_image_list.append(creative_image)
        for steps,name,image_list in zip(
            [args.num_inference_steps, args.num_inference_steps//2, args.num_inference_steps//3],
            ["vanilla","half","third"],
            [vanilla_image_list, half_image_list, third_image_list]
        ):
            generator=torch.Generator().manual_seed(i)
            image=pipeline(prompt,num_inference_steps=steps,generator=generator).images[0]
            path=f"{args.image_dir}/i/{name}.png"
            image.save(path)
            accelerator.log({
                f"{i}/{name}":wandb.Image(path)
            })
            image_list.append(image)
        for name,image_list in zip(
            ["creative","vanilla","half","third"],
            [creative_image_list, vanilla_image_list, half_image_list, third_image_list]
        ):
            print(f"metrics for {name}")
            metric_dict=get_metric_dict(evaluation_prompt_list, image_list, image_list[:1],None)
            for k,v in metric_dict.items():
                print(f"\t{k}: {v}")
                accelerator.log({
                    f"{name}/{k}":v
                })




if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done :)))")