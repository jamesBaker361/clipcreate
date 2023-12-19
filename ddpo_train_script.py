import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
#os.symlink("~/.cache/huggingface/", cache_dir)
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration,BlipModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from torch.nn import Softmax
import torch
import time
from creative_loss import clip_scorer_ddpo
from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
from huggingface_hub import create_repo, upload_folder
from datasets import load_dataset
import random
import argparse
from static_globals import *

def get_prompt_fn(dataset_name,split):
    hf_dataset=load_dataset(dataset_name,split=split)
    prompt_list=[t for t in hf_dataset["text"]]

    def _fn():
        return random.choice(prompt_list),{}
    
    return _fn

parser = argparse.ArgumentParser(description="ddpo training")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default="runwayml/stable-diffusion-v1-5",
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="jlbaker361/wikiart-balanced100",
    help="The name of the Dataset (from the HuggingFace hub) to train on"
)
parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/jlb638/sd-ddpo",
        help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
parser.add_argument(
    "--hub_model_id",
    type=str,
    default=None,
    help="The name of the repository on hf to write to",
)
parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--sample_num_steps",type=int,default=10, help="Number of sampler inference steps per image")
parser.add_argument("--sample_batch_size",type=int,default=4, help="batch size")
parser.add_argument("--train_gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--style_list",nargs="*",help="styles to be used")
parser.add_argument("--sample_num_batches_per_epoch",type=int,default=8)

if __name__=='__main__':
    args = parser.parse_args()
    style_list=len(args.style_list)
    if len(style_list)<2:
        style_list=WIKIART_STYLES
    reward_fn=clip_scorer_ddpo(style_list)
    prompt_fn=get_prompt_fn(args.dataset_name)

    config=DDPOConfig(
        num_epochs=args.num_epochs,
        train_gradient_accumulation_steps=args.train_gradient_accumulation_steps,
        sample_num_steps=args.sample_num_steps,
        sample_batch_size=args.sample_batch_size,
        train_batch_size=args.train_batch_size,
        sample_num_batches_per_epoch=args.sample_num_batches_per_epoch,
        mixed_precision="no"
    )

    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.pretrained_model_name_or_path,  use_lora=True
    )

    trainer = DDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline
    )
    start=time.time()
    trainer.train()
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful training :) time elapsed: {seconds} seconds = {hours} hours")

    trainer._save_pretrained(args.output_dir)
    repo_id = create_repo(repo_id=args.hub_model_id, exist_ok=True).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path=args.output_dir,
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
    )
    print("successful saving :)")
