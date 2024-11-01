#use this to generate the images used for evaluation and tsne reduction

import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.utils import print_trainable_parameters
from experiment_helpers.lora_loading import load_lora_weights,get_pipeline_from_hf,fix_lora_weights
from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from datasets import Dataset,load_dataset
from huggingface_hub import hf_hub_download
import torch
import time
from peft import PeftModel, PeftConfig

parser=argparse.ArgumentParser()


parser.add_argument("--limit",type=int,default=100)
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--output_hf_dataset",type=str,default="evaluation")
parser.add_argument("--hub_model_id",type=str,default="jlbaker361/vanilla")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--prompt_set",type=str,default="all")
parser.add_argument("--prompt_subjects",action="store_true")

prompt_set_dict={
    "mediums":["painting","art","drawing"],
    "subjects":["person","man","woman"],
    "all":["painting","art","drawing","person","man","woman"]
}

prompt_subject_list=[" of a man ", " of a woman "," of nature "," of an animal ", " of a city"," "," of a building "," of a child "]

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    try:
        if args.hub_model_id=="jlbaker361/vanilla":
            pipe=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        else:
            #pipe=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
            pipeline=BetterDefaultDDPOStableDiffusionPipeline("stabilityai/stable-diffusion-2-base",use_lora=True)
            print_trainable_parameters(pipeline.sd_pipeline.unet)
            weight_path=hf_hub_download(repo_id=args.hub_model_id,filename="pytorch_lora_weights.safetensors", repo_type="model")
            #load_weights(pipeline,weight_path,args.adapter_name)
            load_lora_weights(pipeline,weight_path,["weight","default.weight"])
            #print_trainable_parameters(pipeline.sd_pipeline.unet)
            pipe=pipeline.sd_pipeline
    except:
        if args.hub_model_id=="jlbaker361/vanilla":
            pipe=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base",force_download=True)
        else:
            #pipe=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
            pipeline=BetterDefaultDDPOStableDiffusionPipeline("stabilityai/stable-diffusion-2-base",use_lora=True)
            print_trainable_parameters(pipeline.sd_pipeline.unet)
            weight_path=hf_hub_download(repo_id=args.hub_model_id,filename="pytorch_lora_weights.safetensors", repo_type="model")
            #load_weights(pipeline,weight_path,args.adapter_name)
            load_lora_weights(pipeline,weight_path,["weight","default.weight"])
            #print_trainable_parameters(pipeline.sd_pipeline.unet)
            pipe=pipeline.sd_pipeline
    
    '''try:
        pipe.unet=PeftModel.from_pretrained(pipe.unet,args.hub_model_id)
    except:
        fix_lora_weights(args.hub_model_id)
        pipe.unet=PeftModel.from_pretrained(pipe.unet,args.hub_model_id)'''
    print("after loading???")
    print_trainable_parameters(pipe.unet)

    pipe=StableDiffusionPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=pipe.image_encoder,
        requires_safety_checker=False
    )
    

    print("after loading???")
    print_trainable_parameters(pipe.unet)

    
    torch_dtype={
            "no":torch.float16,
            "fp16":torch.float16}[args.mixed_precision]
    pipe.to(accelerator.device)
    pipe.unet,pipe.text_encoder,pipe.vae=accelerator.prepare(pipe.unet,pipe.text_encoder,pipe.vae)
    
    prompt_set=prompt_set_dict[args.prompt_set]
    if args.prompt_subjects:
        _prompt_set=[]
        for p in prompt_set:
            for sub in prompt_subject_list:
                _prompt_set.append(p+sub)
        prompt_set=_prompt_set
    src_dict={
        "image":[],
        "prompt":[],
        "index":[]
    }
    for i in range(args.limit):
        gen=torch.Generator()
        gen.manual_seed(i)
        prompt=prompt_set[i%len(prompt_set)]
        image=pipe(prompt, num_inference_steps=args.num_inference_steps,generator=gen,safety_checker=None).images[0]
        
        src_dict["image"].append(image)
        src_dict["prompt"].append(prompt)
        src_dict["index"].append(i)

        if i%10==0:
            print("finished image ",i)

    Dataset.from_dict(src_dict).push_to_hub(args.output_hf_dataset)
    load_dataset(args.output_hf_dataset)



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