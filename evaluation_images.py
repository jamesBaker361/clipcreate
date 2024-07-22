#use this to generate the images used for evaluation and tsne reduction

import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.lora_loading import load_lora_weights,get_pipeline_from_hf
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from datasets import Dataset,load_dataset
import torch
import time

parser=argparse.ArgumentParser()


parser.add_argument("--limit",type=int,default=100)
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--output_hf_dataset",type=str,default="evaluation")
parser.add_argument("--hub_model_id",type=str,default="jlbaker361/vanilla")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--prompt_set",type=str,default="all")

prompt_set_dict={
    "mediums":["painting","art","drawing"],
    "subjects":["person","man","woman"],
    "all":["painting","art","drawing","person","man","woman"]
}

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    if args.hub_model_id=="jlbaker361/vanilla":
        pipe=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    else:
        pipe=get_pipeline_from_hf(args.hub_model_id,False,False,True,False,use_lora=True,pretrained_model_name="runwayml/stable-diffusion-v1-5").sd_pipeline
    torch_dtype={
            "no":torch.float16,
            "fp16":torch.float16}[args.mixed_precision]
    pipe.to(accelerator.device)
    pipe.unet,pipe.text_encoder,pipe.vae=accelerator.prepare(pipe.unet,pipe.text_encoder,pipe.vae)
    
    prompt_set=prompt_set_dict[args.prompt_set]
    src_dict={
        "image":[],
        "prompt":[],
        "index":[]
    }
    for i in range(args.limit):
        gen=torch.Generator()
        gen.manual_seed(i)
        prompt=prompt_set[i%len(prompt_set)]
        image=pipe(prompt, num_inference_steps=args.num_inference_steps,generator=gen).images[0]
        
        src_dict["image"].append(image)
        src_dict["prompt"].append(prompt)
        src_dict["index"].append(i)

        if i%10==0:
            print("finished image ",i)

    Dataset.from_dict(src_dict).push_to_hub(f"jlbaker361/{args.output_hf_dataset}")
    load_dataset(f"jlbaker361/{args.output_hf_dataset}")



if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")