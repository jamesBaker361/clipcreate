import os
import sys
import torch
if "SLURM_JOB_ID" in os.environ:
    cache_dir="/scratch/jlb638/trans_cache"
    os.environ["TRANSFORMERS_CACHE"]=cache_dir
    os.environ["HF_HOME"]=cache_dir
    os.environ["HF_HUB_CACHE"]=cache_dir

    torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
import argparse
from static_globals import *
import faulthandler
import re
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from huggingface_hub.utils import EntryNotFoundError
from torchvision.transforms.functional import to_pil_image
from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from peft import get_peft_model_state_dict
import torch
import time
from creative_loss import clip_scorer_ddpo, elgammal_dcgan_scorer_ddpo, k_means_scorer
from huggingface_hub import create_repo, upload_folder, ModelCard
from datasets import load_dataset
import random
import numpy as np
import wandb
from diffusers.utils.import_utils import is_xformers_available
from packaging import version

def save_lora_weights(pipeline:BetterDefaultDDPOStableDiffusionPipeline,output_dir:str):
    state_dict=get_peft_model_state_dict(pipeline.sd_pipeline.unet, unwrap_compiled=True)
    weight_path=os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    print("saving to ",weight_path)
    save_file(state_dict, weight_path, metadata={"format": "pt"})

def load_lora_weights(pipeline:BetterDefaultDDPOStableDiffusionPipeline,path:str):
    #pipeline.get_trainable_layers()
    print("loading from ",path)
    state_dict={}
    count=0
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key]=f.get_tensor(key)
    state_dict={
        k.replace("weight","default.weight"):v for k,v in state_dict.items()
    }
    pipeline.sd_pipeline.unet.load_state_dict(state_dict,strict=False)
    print("successfully loaded from")

faulthandler.enable()

def get_prompt_fn(dataset_name,split,unconditional_fraction):
    hf_dataset=load_dataset(dataset_name,split=split)
    prompt_list=[t for t in hf_dataset["text"]]

    def _fn():
        if random.uniform(0.0,1.)<=unconditional_fraction:
            return " ",{}
        return random.choice(prompt_list),{}
    
    return _fn

def get_image_sample_hook(image_dir):
    os.makedirs(image_dir, exist_ok=True)
    def _fn(prompt_image_data, global_step, tracker):
        for row in prompt_image_data:
            images=row[0]
            prompts=row[1]
            for img,pmpt in zip(images, prompts):
                pmpt=pmpt.replace(" ", "_")
                pmpt=re.sub(r'\W+', '', pmpt)
                pmpt=pmpt[:45]
                path=image_dir+pmpt+str(global_step)+".png"
                print("saving at ",path)
                pil_img=to_pil_image(img)
                pil_img.save(path)
                tracker.log({f"{pmpt}":wandb.Image(path)},tracker.tracker.step)
    return _fn

parser = argparse.ArgumentParser(description="ddpo training")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default=None,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)

parser.add_argument("--base_model",type=str,default="stabilityai/stable-diffusion-2-base",help="base model")

parser.add_argument(
    "--dataset_name",
    type=str,
    default="jlbaker361/wikiart-balanced500",
    help="The name of the Dataset (from the HuggingFace hub) to train on"
)
parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/jlb638/sd-ddpo",
        help="The output directory where the model predictions and checkpoints will be written.",
)

parser.add_argument("--image_dir",type=str,default=None)
parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
parser.add_argument(
    "--hub_model_id",
    type=str,
    default=None,
    help="The name of the repository on hf to write to",
)

parser.add_argument("--unconditional_fraction",type=float,default=0.2,help="fraction of prompts to be blank text")
parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--sample_num_steps",type=int,default=10, help="Number of sampler inference steps per image")
parser.add_argument("--sample_batch_size",type=int,default=4, help="batch size for all gpus, must be >= train_batch_size")
parser.add_argument("--train_gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--style_list",nargs="*",help="styles to be used")
parser.add_argument("--sample_num_batches_per_epoch",type=int,default=8)
parser.add_argument("--use_lora",type=bool,default=True)
parser.add_argument("--mixed_precision",type=str, default="no",help="precision, one of no, fp16, bf16")
parser.add_argument("--cache_dir",type=str, default=None)
parser.add_argument("--resume_from",type=str,default=None)
parser.add_argument("--reward_function",default="clip",type=str,help="reward function: resnet dcgan or clip kmeans")
parser.add_argument("--center_list_path",type=str,default="test_centers.npy", help="path for np files that are centers of the clusters for k means")

parser.add_argument("--dcgan_repo_id",type=str,help="hf repo whre dcgan discriminator weights are",default="jlbaker361/dcgan-wikiart1000")
parser.add_argument("--disc_init_dim",type=int,default=32, help="initial layer # of channels in discriminator")
parser.add_argument("--disc_final_dim",type=int, default=512, help="final layer # of channels in discriminator")
parser.add_argument("--resize_dim",type=int,default=512,help="dim to resize images to before cropping (for dcgan)")
parser.add_argument("--image_dim",type=int,default=512,help="image dim for dcgan")
parser.add_argument("--adapter_name",type=str,default="default")


if __name__=='__main__':
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    style_list=args.style_list
    if style_list is None or len(style_list)<2:
        style_list=WIKIART_STYLES
    if args.reward_function == "clip":
        reward_fn=clip_scorer_ddpo(style_list)
    elif args.reward_function=="kmeans":
        reward_fn=k_means_scorer(args.center_list_path)
    elif args.reward_function=="dcgan":
        reward_fn=elgammal_dcgan_scorer_ddpo(style_list,args.image_dim, args.resize_dim, args.disc_init_dim, args.disc_final_dim, args.dcgan_repo_id)
    else:
        raise Exception("unknown reward function; should be one of clip or resnet or dcgan")
    prompt_fn=get_prompt_fn(args.dataset_name, "train",args.unconditional_fraction)
    pipeline=BetterDefaultDDPOStableDiffusionPipeline(args.base_model,use_lora=True)
    if args.pretrained_model_name_or_path is not None:
        try:
            weight_path=hf_hub_download(repo_id=args.pretrained_model_name_or_path,filename="pytorch_lora_weights.safetensors", repo_type="model")
            #load_weights(pipeline,weight_path,args.adapter_name)
            load_lora_weights(pipeline,weight_path)
            print(f"loaded weights from {args.pretrained_model_name_or_path}")
        except:
            print(f"couldn't load lora weights from {args.pretrained_model_name_or_path}")

    resume_from_path=None
    if args.output_dir is not None:
        os.makedirs(args.output_dir,exist_ok=True)
    if args.resume_from is not None:
        os.makedirs(args.resume_from,exist_ok=True)
    start_epoch=0
    if args.resume_from:
        resume_from_path = os.path.normpath(os.path.expanduser(args.resume_from))
        if os.path.exists(resume_from_path):
            checkpoints = list(
                filter(
                    lambda x: "checkpoint_" in x,
                    os.listdir(resume_from_path),
                )
            )
            if len(checkpoints) != 0:
                checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
                resume_from_path = os.path.join(
                    resume_from_path,
                    f"checkpoint_{checkpoint_numbers[-1]}",
                )
                weight_path=os.path.join(resume_from_path, "pytorch_lora_weights.safetensors")
                #load_weights(pipeline,weight_path,args.adapter_name)
                load_lora_weights(pipeline,weight_path)
                start_epoch = checkpoint_numbers[-1] + 1
    start=time.time()
    if args.image_dir==None:
        args.image_dir="images"
        os.makedirs(args.image_dir, exist_ok=True)
    if is_xformers_available():
        print("xformers?!?!")
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            print("xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
        #pipeline.sd_pipeline.unet.enable_xformers_memory_efficient_attention()
    else:
        print("xformers is not available. Make sure it is installed correctly")
    image_samples_hook=get_image_sample_hook(args.image_dir)
    for e in range(start_epoch,args.num_epochs):
        config=DDPOConfig(
            num_epochs=1,
            train_gradient_accumulation_steps=args.train_gradient_accumulation_steps,
            sample_num_steps=args.sample_num_steps,
            sample_batch_size=args.sample_batch_size,
            train_batch_size=args.train_batch_size,
            sample_num_batches_per_epoch=args.sample_num_batches_per_epoch,
            mixed_precision=args.mixed_precision,
            tracker_project_name="ddpo",
            log_with="wandb",
            accelerator_kwargs={
                #"project_dir":args.output_dir
            },
            #project_kwargs=project_kwargs
        )
        trainer = DDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline,
            image_samples_hook
        )
        trainer.train()
        save_lora_weights(pipeline, args.output_dir)
        checkpoint=os.path.join(args.output_dir, f"checkpoint_{e}")
        os.makedirs(checkpoint,exist_ok=True)
        save_lora_weights(pipeline, checkpoint)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful training :) time elapsed: {seconds} seconds = {hours} hours")

    #trainer._save_pretrained(args.output_dir)
    repo_id = create_repo(repo_id=args.hub_model_id, exist_ok=True).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path=args.output_dir,
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
    )
    model_card_content=f"""
    # DDPO trained model
    num_epochs={args.num_epochs} \n
    train_gradient_accumulation_steps={args.train_gradient_accumulation_steps} \n
    sample_num_steps={args.sample_num_steps} \n
    sample_batch_size={args.sample_batch_size} \n 
    train_batch_size={args.train_batch_size} \n
    sample_num_batches_per_epoch={args.sample_num_batches_per_epoch} \n
    based off of stabilityai/stable-diffusion-2-base
    and then trained off of {args.pretrained_model_name_or_path}
    """
    card=ModelCard(model_card_content)
    card.push_to_hub(repo_id)
    print("successful saving :)")
