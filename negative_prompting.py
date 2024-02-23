import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
import torchvision.transforms as transforms
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from torchmetrics.image.inception import InceptionScore
from accelerate import Accelerator
from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
import numpy as np

import wandb
from static_globals import *
from datasets import Dataset
from huggingface_hub import upload_file
torch.manual_seed(0)
import argparse

parser=argparse.ArgumentParser("negative prompting")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--dir",type=str,default="images")
parser.add_argument("--n_images",type=int,default=10)
parser.add_argument("--prompt",type=str,default=" ")
parser.add_argument("--base_model",type=str, default="stabilityai/stable-diffusion-2-base")
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--repo_id",type=str,default="jlbaker361/negative_creativity")

#lora_generator= torch.Generator(device="cpu").manual_seed(0)

from call_neg import call_multi_neg, call_vanilla
def main(args):
    aesthetic_fn=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)
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

    NEGATIVE="negative_image"
    VANILLA="vanilla_image"
    NEGATIVE_SCORE=NEGATIVE+"_score"
    VANILLA_SCORE=VANILLA+"_score"
    src_dict={
        NEGATIVE:[],
        VANILLA:[],
        VANILLA_SCORE:[],
        NEGATIVE_SCORE:[]
    }
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    neg_generator= torch.Generator(device="cpu").manual_seed(args.seed)


    for x in range(args.n_images):
        neg_img=call_multi_neg(pipeline.sd_pipeline,"painting",num_inference_steps=args.num_inference_steps,
                               generator=neg_generator,
                        negative_prompt=["horse","donkey"]).images[0]
        path=f"{args.dir}/img{x}.png"
        neg_img.save(path)
        neg_img_score=aesthetic_fn(neg_img,{},{})[0]

        accelerator.log({NEGATIVE: wandb.Image(path)},step=x)
        accelerator.log({NEGATIVE_SCORE:neg_img_score},step=x)
        src_dict[NEGATIVE].append(neg_img)
        src_dict[NEGATIVE_SCORE].append(neg_img_score)

        vanilla_img=pipeline("painting",num_inference_steps=args.num_inference_steps,generator=generator).images[0]
        path=f"{args.dir}/vanilla_img{x}.png"
        vanilla_img.save(path)
        vanilla_img_score=aesthetic_fn(vanilla_img,{},{})[0]

        accelerator.log({VANILLA: wandb.Image(path)},step=x)
        accelerator.log({VANILLA_SCORE: vanilla_img_score},step=x)
        src_dict[VANILLA].append(vanilla_img)
        src_dict[VANILLA_SCORE].append(vanilla_img_score)

    run_url=accelerator.get_tracker("wandb").run.get_url()
    Dataset.from_dict(src_dict).push_to_hub(args.repo_id)
    model_card_content=f"created a total of {args.n_images} images \n\n"
    model_card_content+=f"wandb run url: {run_url}\n\n"
    for key in [NEGATIVE,VANILLA]:
        inception = InceptionScore(normalize=True)
        tensor_stack=torch.stack([transforms.PILToTensor()(img) for img in  src_dict[key]])
        inception.update(tensor_stack)
        inception_mean, inception_std=inception.compute()
        model_card_content+=f"{key} inception mean: {inception_mean} std: {inception_std}\n\n"
        score_list=src_dict[key+"_score"]
        ava_std=np.std(score_list)
        ava_mean=np.mean(score_list)
        model_card_content+=f"{key} ava mean: {ava_mean} std: {ava_std}\n\n"
    with open("tmp_neg.md","w+") as file:
        file.write(model_card_content)
    upload_file(path_or_fileobj="tmp_neg.md", 
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset")

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    main(args)