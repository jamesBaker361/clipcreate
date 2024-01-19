import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
from huggingface_hub.utils import EntryNotFoundError
#os.symlink("~/.cache/huggingface/", cache_dir)
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
import torch
import wandb
from datasets import Dataset,load_dataset
from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
import random
import argparse
from static_globals import *
import numpy as np

random.seed(1234)

parser = argparse.ArgumentParser(description="evaluation")

parser.add_argument(
    "--model_list",
    type=str,
    nargs="*",
    help="Path to pretrained models or model identifiers from huggingface.co/models.",
)

parser.add_argument(
    "--dataset_name",
    type=str,
    default="jlbaker361/wikiart-balanced100",
    help="The name of the Dataset (from the HuggingFace hub) to train on"
)

parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=30
)

parser.add_argument("--hf_dir",type=str,default="jlbaker361/evaluation",help="hf dir to push to")

parser.add_argument("--image_root_dir",type=str,default="/scratch/jlb638/evaluation_images/")
parser.add_argument("--limit",type=int,default=150,  help="how many samples to make")

if __name__=='__main__':
    args = parser.parse_args()
    print(args)
    aesthetic_fn=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)
    src_dict={
        "prompt":[],
        "image":[],
        "model":[],
        "score":[],
        "name":[]
    }

    hf_dataset=load_dataset(args.dataset_name,split="test")
    prompt_list=[[t,n] for t,n in zip(hf_dataset["text"], hf_dataset["name"])]
    random.shuffle(prompt_list)
    prompt_list=prompt_list[:args.limit]
    model_dict={}
    for model in args.model_list:
        try:
            pipeline=DefaultDDPOStableDiffusionPipeline(model, use_lora=True)
            print(f"loaded weights for {model}")
        except (EntryNotFoundError,ValueError) as error:
            print(error)
            pipeline=DefaultDDPOStableDiffusionPipeline("runwayml/stable-diffusion-v1-5")
            try:
                pipeline.sd_pipeline.load_lora_weights(model,weight_name="pytorch_lora_weights.safetensors")
            except Exception as error:
                print(error)
            try:
                slurm_job_id=os.environ["SLURM_JOB_ID"]
                with open(f"slurm/out/{slurm_job_id}.out","a+") as file:
                    print(f"\nloaded lora weights spearately for {model} SLURM_JOB_ID={slurm_job_id}",file=file)
            except:
                print(f"loaded lora weights spearately for {model}")
        #pipeline.set_progress_bar_config(disable=True)
        model_dict[model]=pipeline

    table_data=[]
    columns=["image","model","prompt","score"]
    
    for model,pipeline in model_dict.items():
        
        total_score=0.0
        score_list=[]
        for [prompt,name] in prompt_list:
            image = pipeline(prompt, num_inference_steps=args.num_inference_steps).images[0]
            src_dict["prompt"].append(prompt)
            src_dict["image"].append(image)
            src_dict["model"].append(model)
            score=aesthetic_fn(image,{},{})[0].numpy()[0]
            src_dict["score"].append(score)
            src_dict["name"].append(name)
            #table_data.append([wandb.Image(image), model, prompt, score])
            total_score+=score
            score_list.append(score)
            try:
                slurm_job_id=os.environ["SLURM_JOB_ID"]
                with open(f"slurm/out/{slurm_job_id}.out","a+") as file:
                    print("\n",prompt,score,file=file)
            except:
                print(prompt,score)
        score_std=np.std(score_list)
        try:
            slurm_job_id=os.environ["SLURM_JOB_ID"]
            with open(f"slurm/out/{slurm_job_id}.out","a+") as file:
                print(f"\ntotal score {model} {total_score} std {score_std}",file=file)
        except:
            print(f"total score {model} {total_score} std {score_std}")

    #run = wandb.init(project="creative_clip")
    #table=wandb.Table(columns=columns,data=table_data)
    #run.log({"table":table})
    Dataset.from_dict(src_dict).push_to_hub(args.hf_dir)
    try:
        slurm_job_id=os.environ["SLURM_JOB_ID"]
        with open(f"slurm/out/{slurm_job_id}.out","a+") as file:
            print("all done :)))",file=file)
    except:
        print("all done :)))")