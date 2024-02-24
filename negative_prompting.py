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
parser.add_argument("--lora_model",type=str,default="jlbaker361/ddpo-stability-e5")
parser.add_argument("--lora_model_dcgan",type=str,default="jlbaker361/ddpo-stability-dcgan-e5")
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--repo_id",type=str,default="jlbaker361/negative_creativity")
parser.add_argument("--index",type=int,default=0)

#lora_generator= torch.Generator(device="cpu").manual_seed(0)

from call_neg import call_multi_neg, call_vanilla
def main(args):
    aesthetic_fn=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)
    
    run=wandb.init(project="lora scale comparison")

    pipeline=DefaultDDPOStableDiffusionPipeline(args.base_model)
    lora_pipeline=DefaultDDPOStableDiffusionPipeline(args.base_model)
    lora_pipeline.sd_pipeline.load_lora_weights(args.lora_model,weight_name="pytorch_lora_weights.safetensors")
    lora_dcgan_pipeline=DefaultDDPOStableDiffusionPipeline(args.base_model)
    lora_dcgan_pipeline.sd_pipeline.load_lora_weights(args.lora_model_dcgan,weight_name="pytorch_lora_weights.safetensors")
    #pipeline.sd_pipeline.to(device)
    #pipeline.sd_pipeline=accelerator.prepare(pipeline.sd_pipeline)
    #generator = torch.Generator(device=accelerator.device).manual_seed(0)

    NEGATIVE="negative_image"
    VANILLA="vanilla_image"
    SIMPLE_NEGATIVE="simple_negative_image"
    NEGATIVE_SCORE=NEGATIVE+"_score"
    VANILLA_SCORE=VANILLA+"_score"
    SIMPLE_NEGATIVE_SCORE=SIMPLE_NEGATIVE+"_score"
    SCALE="scaled_lora"
    SCALE_SCORE="scaled_lora_score"
    LORA="lora"
    LORA_SCORE="lora_score"
    SCALE_DCGAN="scaled_lora_dcgan"
    SCALE_DCGAN_SCORE=SCALE_DCGAN+"_score"
    LORA_DCGAN="lora_dcgan"
    LORA_DCGAN_SCORE=LORA_DCGAN+"_score"
    VANILLA_HALF="vanilla_half"
    VANILLA_HALF_SCORE="vanilla_half_score"
    src_dict={
        NEGATIVE:[],
        VANILLA:[],
        VANILLA_SCORE:[],
        NEGATIVE_SCORE:[],
        SIMPLE_NEGATIVE:[],
        SIMPLE_NEGATIVE_SCORE:[],
        SCALE:[],
        SCALE_SCORE:[],
        SCALE_DCGAN:[],
        SCALE_DCGAN_SCORE:[],
        LORA_DCGAN:[],
        LORA_DCGAN_SCORE:[],
        LORA:[],
        LORA_SCORE:[],
        VANILLA_HALF:[],
        VANILLA_HALF_SCORE:[]
    }
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    half_generator=torch.Generator(device="cpu").manual_seed(args.seed)
    neg_generator= torch.Generator(device="cpu").manual_seed(args.seed)
    simple_generator=torch.Generator(device="cpu").manual_seed(args.seed)
    lora_scale_generator=torch.Generator(device="cpu").manual_seed(args.seed)
    lora_generator= torch.Generator(device="cpu").manual_seed(args.seed)
    lora_dcgan_generator= torch.Generator(device="cpu").manual_seed(args.seed)
    lora_scale_dcgan_generator= torch.Generator(device="cpu").manual_seed(args.seed)

    #WIKIART_STYLES=["cat","dog"]
    for n in range(args.index,args.n_images+args.index):
        gen_state=generator.get_state()
        half_generator.set_state(gen_state) #we want the half generator to start at the same place as the normal gen so initial N(0,1) is the same

        neg_img=call_multi_neg(pipeline.sd_pipeline,args.prompt,num_inference_steps=args.num_inference_steps,
                               generator=neg_generator,
                        negative_prompt=WIKIART_STYLES).images[0]
        neg_path=f"{args.dir}/img{n}.png"

        img_vanilla=pipeline(args.prompt,num_inference_steps=args.num_inference_steps,generator=generator).images[0]
        vanilla_path=f"{args.dir}/vanilla_img{n}.png"

        img_lora=lora_pipeline(args.prompt, num_inference_steps=args.num_inference_steps,generator=lora_generator).images[0]
        lora_path=f"{args.dir}/img{n}_lora.jpg"

        lora_scale_path=f"{args.dir}/img{n}_lora_scaled.jpg"
        img_lora_scale=lora_pipeline(args.prompt, num_inference_steps=args.num_inference_steps,generator=lora_scale_generator,cross_attention_kwargs={"scale": 3.0}).images[0]

        negative_prompt=",".join(WIKIART_STYLES)
        simple_neg_img=pipeline(args.prompt,num_inference_steps=args.num_inference_steps,generator=simple_generator,
                                negative_prompt=negative_prompt).images[0]
        simple_neg_path=f"{args.dir}/simple_neg_img{n}.png"

        half_path=f"{args.dir}/img{n}_half.jpg"
        img_half=pipeline(args.prompt,num_inference_steps=args.num_inference_steps//2,generator=half_generator).images[0]


        img_lora_dcgan=lora_dcgan_pipeline(args.prompt, num_inference_steps=args.num_inference_steps,generator=lora_dcgan_generator).images[0]
        lora_dcgan_path=f"{args.dir}/img{n}_dcgan_lora.jpg"

        lora_scale_dcgan_path=f"{args.dir}/img{n}_dcgan_lora_scaled.jpg"
        img_lora_scale_dcgan=lora_dcgan_pipeline(args.prompt, num_inference_steps=args.num_inference_steps,generator=lora_scale_dcgan_generator,cross_attention_kwargs={"scale": 3.0}).images[0]

        for key,path,img in zip(
            [VANILLA,LORA,SCALE,VANILLA_HALF, NEGATIVE, SIMPLE_NEGATIVE,LORA_DCGAN, SCALE_DCGAN],
            [vanilla_path,lora_path, lora_scale_path, half_path, neg_path, simple_neg_path, lora_dcgan_path, lora_scale_dcgan_path],
            [img_vanilla, img_lora,img_lora_scale, img_half, neg_img, simple_neg_img,img_lora_dcgan, img_lora_scale_dcgan]
        ):
            img.save(path)
            score_key=key+"_score"
            score=aesthetic_fn(img,{},{})[0]
            run.log({key:wandb.Image(path)},step=n)
            src_dict[key].append(img)
            run.log({score_key: score},step=n)
            src_dict[score_key].append(score)    

    run_url=run.get_url()
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