import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
import torchvision.transforms as transforms
import torch
import wandb
import argparse
from datasets import Dataset
from huggingface_hub import upload_file

parser = argparse.ArgumentParser(description="comparison of lora scales")
parser.add_argument("--seed",type=int,default=0, help="random seed")
parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2-base")
parser.add_argument("--lora_model",type=str,default="jlbaker361/ddpo-stability-e5")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--n_images",type=int,default=30)
parser.add_argument("--dir",type=str,default="lora_scale_comp")
parser.add_argument("--repo_id",type=str,default="jlbaker361/lora_scale_comparison")

def main(args):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    lora_generator= torch.Generator(device="cpu").manual_seed(args.seed)
    lora_scale_generator=torch.Generator(device="cpu").manual_seed(args.seed)

    pipeline=DefaultDDPOStableDiffusionPipeline(args.base_model)
    lora_pipeline=DefaultDDPOStableDiffusionPipeline(args.base_model)
    lora_pipeline.sd_pipeline.load_lora_weights(args.lora_model,weight_name="pytorch_lora_weights.safetensors")

    os.makedirs(args.dir,exist_ok=True,mode=777)

    SCALE="scaled_lora"
    LORA="lora"
    VANILLA="vanilla"

    src_dict={
        SCALE:[],
        LORA:[],
        VANILLA:[]
    }

    run=wandb.init(project="lora scale comparison")

    for n in range(args.n_images):
        img_vanilla=pipeline(" ",num_inference_steps=args.num_inference_steps,generator=generator).images[0]
        vanilla_path=f"{args.dir}/img{n}.jpg"
        img_vanilla.save(vanilla_path)
        img_lora=lora_pipeline(" ", num_inference_steps=args.num_inference_steps,generator=lora_generator).images[0]
        lora_path=f"{args.dir}/img{n}_lora.jpg"
        img_lora.save(lora_path)
        lora_scale_path=f"{args.dir}/img{n}_lora_scaled.jpg"
        img_lora_scale=lora_pipeline(" ", num_inference_steps=args.num_inference_steps,generator=lora_scale_generator,cross_attention_kwargs={"scale": 3.0}).images[0]
        img_lora_scale.save(lora_scale_path)

        run.log({VANILLA: wandb.Image(vanilla_path)},step=n)
        run.log({LORA: wandb.Image(lora_path)},step=n)
        run.log({SCALE: wandb.Image(lora_scale_path) },step=n)

        src_dict[VANILLA].append(img_vanilla)
        src_dict[LORA].append(img_lora)
        src_dict[SCALE].append(img_lora_scale)
    Dataset.from_dict(src_dict).push_to_hub(args.repo_id)
    model_card_content=f"created a total of {args.n_images} images \n"
    model_card_content+=f"wandb run url: {run.get_url()}\n"
    with open("tmp_lora_scale.md","w+") as file:
        file.write(model_card_content)
    upload_file(path_or_fileobj="tmp_lora_scale.md", 
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset")
    

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    main(args)