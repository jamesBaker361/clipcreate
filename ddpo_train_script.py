import os
import sys
import torch
import argparse
from static_globals import *
import faulthandler
import re
from safetensors import safe_open
from safetensors.torch import save_file
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from trl import DDPOConfig
from better_ddpo_trainer import BetterDDPOTrainer
from torchvision.transforms.functional import to_pil_image
from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from peft import get_peft_model_state_dict
import torch
import time
from creative_loss import clip_scorer_ddpo, elgammal_dcgan_scorer_ddpo, k_means_scorer,image_reward_scorer
import random
import numpy as np
import wandb
from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
import datetime
import PIL
from huggingface_hub import HfApi,snapshot_download

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
    param_set=set([p[0] for p in pipeline.sd_pipeline.unet.named_parameters()])
    for k in state_dict.keys():
        if k in param_set:
            count+=1
    print(f"loaded {count} params")
    pipeline.sd_pipeline.unet.load_state_dict(state_dict,strict=False)
    print("successfully loaded from")

faulthandler.enable()

def get_prompt_fn(dataset_name,split,unconditional_fraction,text_col_name):
    '''try:
        hf_dataset=load_dataset(dataset_name,split=split)
    except:
        time.sleep(120.0)
        hf_dataset=load_dataset(dataset_name,split=split)
    prompt_list=[t for t in hf_dataset[text_col_name]]
    print("prompt list len",len(prompt_list))'''
    medium_list=["painting of ","picture of ","drawing of "]
    subject_list=["a man"," a woman"," a landscape"," nature"," a building"," an animal"," shapes", " an object"]

    def _fn():
        if random.uniform(0.0,1.)<=unconditional_fraction:
            return " ",{}
        return random.choice(medium_list)+random.choice(subject_list),{}
    
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
                pil_img=to_pil_image(img)
                pil_img.save(path)
                print("saved at ",path)
                try:
                    tracker.log({f"{pmpt}":wandb.Image(path)})
                except (PIL.UnidentifiedImageError,OSError):
                    try:
                        tracker.log({f"{pmpt}":wandb.Image(pil_img)})
                    except Exception as err:
                        print("couldnt log image??? because", err)
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

parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/ddpo-images/")
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
parser.add_argument("--resume_from",type=str,default=None)
parser.add_argument("--reward_function",default="clip",type=str,help="reward function: resnet dcgan or clip kmeans")
parser.add_argument("--center_list_path",type=str,default="test_centers.npy", help="path for np files that are centers of the clusters for k means")

parser.add_argument("--dcgan_repo_id",type=str,help="hf repo whre dcgan discriminator weights are",default="jlbaker361/dcgan-vanilla")
parser.add_argument("--text_col_name",type=str,default="text",help="name of text col for captions in dataset")
parser.add_argument("--disc_init_dim",type=int,default=32, help="initial layer # of channels in discriminator")
parser.add_argument("--disc_final_dim",type=int, default=512, help="final layer # of channels in discriminator")
parser.add_argument("--resize_dim",type=int,default=512,help="dim to resize images to before cropping (for dcgan)")
parser.add_argument("--image_dim",type=int,default=512,help="image dim for dcgan")
parser.add_argument("--adapter_name",type=str,default="default")
parser.add_argument("--project_name",type=str,default="ddpo-creativity")


if __name__=='__main__':
    for slurm_var in ["SLURMD_NODENAME","SBATCH_CLUSTERS", 
                      "SBATCH_PARTITION","SLURM_JOB_PARTITION",
                      "SLURM_NODEID","SLURM_MEM_PER_GPU",
                      "SLURM_MEM_PER_CPU","SLURM_MEM_PER_NODE","SLURM_JOB_ID"]:
        try:
            print(slurm_var, os.environ[slurm_var])
        except:
            print(slurm_var, "doesnt exist")
    args = parser.parse_args()
    print(args)
    current_date_time = datetime.datetime.now()
    formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Formatted Date and Time:", formatted_date_time)
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
    elif args.reward_function=="aesthetic":
        reward_fn=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)
    elif args.reward_function=="image_reward":
        reward_fn =image_reward_scorer()
    else:
        raise Exception("unknown reward function; should be one of clip or resnet or dcgan")
    prompt_fn=get_prompt_fn(args.dataset_name, "train",args.unconditional_fraction,args.text_col_name)
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
                checkpoint_numbers=[c for c in checkpoint_numbers if c<=args.num_epochs]
                if len(checkpoint_numbers)>0:
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
    image_samples_hook=get_image_sample_hook(args.image_dir)
    accelerator=Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
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
        trainer = BetterDDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline,
            image_samples_hook
            ,image_dim=args.image_dim
        )
        print(f"acceleerate device {trainer.accelerator.device}")
        tracker=trainer.accelerator.get_tracker("wandb").run
        trainer.train()
        save_lora_weights(pipeline, args.output_dir)
        if e%5==0:
            checkpoint=os.path.join(args.output_dir, f"checkpoint_{e}")
            os.makedirs(checkpoint,exist_ok=True)
            save_lora_weights(pipeline, checkpoint)
        validation_prompt_list=["painting of a man","picture of nature"]
        for validation_prompt in validation_prompt_list:
            generator=torch.Generator(trainer.accelerator.device)
            generator.manual_seed(123)
            validation_image=pipeline(validation_prompt,num_inference_steps=args.sample_num_steps,generator=generator,height=args.image_dim,width=args.image_dim).images[0]
            validation_prompt=validation_prompt.replace(" ","_")
            validation_path=f"{args.image_dir}{validation_prompt}.png"
            validation_image.save(validation_path)
            try:
                tracker.log({f"{validation_prompt}":wandb.Image(validation_path)},tracker.step)
            except PIL.UnidentifiedImageError:
                pass

    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful training :) time elapsed: {seconds} seconds = {hours} hours")
    api = HfApi()
    api.upload_folder(
    folder_path=args.output_dir,
    repo_id=args.hub_model_id,
    repo_type="model",
    )
    snapshot_download(args.hub_model_id,"model")
    print("successful saving :)")
