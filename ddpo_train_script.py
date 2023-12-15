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
from creative_loss import clip_scorer_ddpo
from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
from huggingface_hub import create_repo, upload_folder
from datasets import load_dataset
import random
import argparse

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
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help=(
        "The name of the Dataset (from the HuggingFace hub) to train on"
    ),
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
    help="The name of the repository on hf",
)
parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
)
parser.add_argument("--num_train_epochs", type=int, default=10)
parser.add_argument("--sample_num_steps",type=int,default=10, help="Number of sampler inference steps.")
parser.add_argument("--sample_batch_size",type=int,default=10, help="batch size")
parser.add_argument("--train_batch_size",type=int,default=10, help="actual batch size???")


train_gradient_accumulation_steps=1,
                sample_num_steps=2,
                sample_batch_size=2,
                train_batch_size=1,
                sample_num_batches_per_epoch=2,