import os
from accelerate import Accelerator
from experiment_helpers.measuring import get_metric_dict
from experiment_helpers.static_globals import AESTHETIC_SCORE,IMAGE_REWARD,PROMPT_SIMILARITY

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

parser = argparse.ArgumentParser(description="evaluation")

parser.add_argument("--model",type=str,default="jlbaker361/dcgan-vanilla")
parser.add_argument("--limit",type=int,default=100,  help="how many samples to make")
parser.add_argument("--project_name",type=str,default="ddpo-evaluation")
parser.add_argument("--seed",type=int,default=123)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/ddpo-eval-images/")
parser.add_argument("--num_inference_steps",type=int,default=30)

def evaluate(args):
    os.makedirs(args.image_dir,exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator=Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name=args.project_name, config=vars(args))

    prompt_fn=get_prompt_fn(None, None,0.1,None)

    generator=torch.Generator(accelerator.device).manual_seed(args.seed)
    pipeline=BetterDefaultDDPOStableDiffusionPipeline("stabilityai/stable-diffusion-2-base")
    weight_path=hf_hub_download(repo_id=args.model, filename="pytorch_lora_weights.safetensors",repo_type="model")
    load_lora_weights(pipeline,weight_path)
    unet=pipeline.sd_pipeline.unet
    vae=pipeline.sd_pipeline.vae
    text_encoder=pipeline.sd_pipeline.text_encoder
    for model in [unet,text_encoder,vae]:
        model.eval()
        model.to(accelerator.device)
    unet,text_encoder,vae=accelerator.prepare(unet,text_encoder,vae)
    model_name=args.model.split("/")[:1]
    os.makedirs(os.path.join(args.image_dir, model_name),exist_ok=True)
    evaluation_prompt_list=[prompt_fn()[0] for _ in range(args.limit)]
    evaluation_image_list=[
        pipeline(prompt, num_inference_steps=args.num_inference_steps,generator=generator).images[0].resize((512,512)) for prompt in evaluation_prompt_list
    ]
    for i,(prompt,image) in enumerate(zip(evaluation_prompt_list, evaluation_image_list)):
        unique_path=f"_{i}_"+prompt.replace(" ","_")[:50]+"_.png"
        path=os.path.join(args.image_dir, model_name,unique_path)
        image.save(path)
        accelerator.log({
            path:wandb.Image(path)
        })
    accelerator.free_memory()
    torch.cuda.empty_cache()
    del unet,text_encoder,vae,pipeline
    accelerator.free_memory()
    torch.cuda.empty_cache()
    metric_dict=get_metric_dict(evaluation_prompt_list, evaluation_image_list, evaluation_image_list[:1],accelerator)
    for metric in [AESTHETIC_SCORE, IMAGE_REWARD,PROMPT_SIMILARITY]:
        accelerator.log({
            metric:metric_dict[metric]
        })
        print(metric, metric_dict[metric])



if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    evaluate(args)
    print("all done")