import os
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
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
import random

parser = argparse.ArgumentParser(description="appendix images")

parser.add_argument("--limit",type=int,default=5,  help="how many samples to make")
parser.add_argument("--project_name",type=str,default="ddpo-appendix")
parser.add_argument("--seed",type=int,default=123)
parser.add_argument("--image_dir",type=str,default="/ddpo-appendix-images/")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--model_list",nargs="*")

def get_pipeline(model,device):
    pipeline=BetterDefaultDDPOStableDiffusionPipeline("stabilityai/stable-diffusion-2-base")
    weight_path=hf_hub_download(repo_id=model, filename="pytorch_lora_weights.safetensors",repo_type="model")
    load_lora_weights(pipeline,weight_path)
    pipeline.sd_pipeline.unet.to(device)
    pipeline.sd_pipeline.text_encoder.to(device)
    pipeline.sd_pipeline.vae.to(device)
    return pipeline

def main(args):
    medium_list=["painting of ","picture of ","drawing of "]
    subject_list=["a man"," a woman"," a landscape"," nature"," a building"," an animal"," shapes", " an object"]
    prompt_list=[]
    for medium in medium_list:
        for subject in subject_list:
            prompt=medium+subject
            prompt_list.append(prompt)
        prompt_list.append(" ")
    random.shuffle(prompt_list)
    accelerator=Accelerator(log_with="wandb")
    accelerator.init_trackers(args.project_name,config=vars(args))
    model_dict={
    model: get_pipeline(model,accelerator.device) for model in args.model_list
    }
    image_dir=os.path.join("/scratch/jlb638",args.image_dir)
    top=" & ".join([model for model in model_dict.keys()])
    print(top," \\\\")

    for x in range(args.limit):
        os.makedirs(image_dir,exist_ok=True)
        prompt=prompt_list[x %(len(prompt_list))]
        print(prompt)
        clean_prompt=prompt.replace(" ","_")
        table_str=f"{prompt} "
        for model_name, pipeline in model_dict.items():
            model_id=model_name.split("/")[1]
            generator=torch.Generator(accelerator.device).manual_seed(args.seed)
            image=pipeline(prompt, num_inference_steps=args.num_inference_steps,generator=generator).images[0].resize((512,512))
            file_path=f"{model_id}_{x}.png"
            path=os.path.join(image_dir,file_path)
            image.save(path)
            table_str+=f" & {os.path.join(args.image_dir,file_path)}"
            accelerator.log({
                f"{x}/{clean_prompt}":wandb.Image(path)
            })
        table_str+=" \\\\"
        print(table_str)






if __name__=='__main__':
    for slurm_var in ["SLURMD_NODENAME","SBATCH_CLUSTERS", 
                      "SBATCH_PARTITION","SLURM_JOB_PARTITION",
                      "SLURM_NODEID","SLURM_MEM_PER_GPU",
                      "SLURM_MEM_PER_CPU","SLURM_MEM_PER_NODE","SLURM_JOB_ID"]:
        try:
            print(slurm_var, os.environ[slurm_var])
        except:
            print(slurm_var, "doesnt exist")
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done :)))")