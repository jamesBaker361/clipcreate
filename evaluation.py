import os
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from huggingface_hub.utils import EntryNotFoundError
from huggingface_hub import upload_file, hf_hub_download
#os.symlink("~/.cache/huggingface/", cache_dir)
from trl import DefaultDDPOStableDiffusionPipeline
from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from ddpo_train_script import load_lora_weights
from datasets import Dataset,load_dataset
from torchvision.transforms import PILToTensor
from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
import random
import argparse
from static_globals import *
import numpy as np
from torchmetrics.image.inception import InceptionScore

random.seed(1234)

parser = argparse.ArgumentParser(description="evaluation")

parser.add_argument(
    "--model_list",
    type=str,
    nargs="*",
    help="Path to pretrained models or model identifiers from huggingface.co/models.",
)

parser.add_argument(
    "--conditional_model_list",
    type=str,
    nargs="*",
    help="Path to pretrained models or model identifiers from huggingface.co/models where conditional==True",
)

parser.add_argument(
    "--dataset_name",
    type=str,
    default="jlbaker361/wikiart-balanced100",
    help="The name of the Dataset (from the HuggingFace hub) to train on"
)

parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=30
)

parser.add_argument("--seed",type=int,default=0)

parser.add_argument("--hf_dir",type=str,default="jlbaker361/evaluation",help="hf dir to push to")

parser.add_argument("--image_root_dir",type=str,default="/scratch/jlb638/evaluation_images/")
parser.add_argument("--limit",type=int,default=150,  help="how many samples to make")
parser.add_argument("--lora_scale",type=float,help="lora scale 0-1 0 is the same as not using the LoRA weights, whereas 1 means only the LoRA fine-tuned weights will be used")
parser.add_argument("--subfolder",type=str)

if __name__=='__main__':
    args = parser.parse_args()
    print(args)
    aesthetic_fn=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)
    src_dict={
        "prompt":[],
        "image":[],
        "model":[],
        "score":[],
        "name":[]
    }
    if args.conditional_model_list is None:
        args.conditional_model_list=[]
    if args.model_list is None:
        args.model_list=[]

    #args.conditional_model_list=[f"{c}-CONDTIONAL" for c in args.conditional_model_list]

    hf_dataset=load_dataset(args.dataset_name,split="test")
    prompt_list=[[t,n] for t,n in zip(hf_dataset["text"], hf_dataset["name"])]
    random.shuffle(prompt_list)
    prompt_list=prompt_list[:args.limit]
    model_dict={}
    gen_dict={}
    inception = InceptionScore(normalize=True)
    for model in args.conditional_model_list+args.model_list:
        pipeline=BetterDefaultDDPOStableDiffusionPipeline("stabilityai/stable-diffusion-2-base")
        try:
            if args.subfolder==None:
                weight_path=hf_hub_download(repo_id=model, filename="pytorch_lora_weights.safetensors",repo_type="model")
            else:
                weight_path=hf_hub_download(repo_id=model, subfolder=args.subfolder, filename="pytorch_lora_weights.safetensors",repo_type="model")
            #load_weights(pipeline,weight_path,args.adapter_name)
            load_lora_weights(pipeline,weight_path)
            print(f"loaded weights from {model}")
        except:
            print(f"couldn't load lora weights from {model}")
        #pipeline.set_progress_bar_config(disable=True)
        if model in args.conditional_model_list:
            generator=torch.Generator(device="cpu").manual_seed(args.seed)
            model_dict[model+"-CONDITIONAL"]=(pipeline,generator)
        generator=torch.Generator(device="cpu").manual_seed(args.seed)
        model_dict[model]=(pipeline,generator)


    table_data=[]
    columns=["image","model","prompt","score"]
    result_dict={}
    for model,(pipeline,generator) in model_dict.items():
        result_dict[model]={}
        total_score=0.0
        score_list=[]
        image_list=[]
        for [prompt,name] in prompt_list:
            if model.find("-CONDITIONAL")==-1:
                prompt=""
            if args.lora_scale is None:
                image = pipeline(prompt, num_inference_steps=args.num_inference_steps,generator=generator).images[0]
            else:
                image = pipeline(prompt, num_inference_steps=args.num_inference_steps, generator=generator,cross_attention_kwargs={"scale": args.lora_scale}).images[0]
            src_dict["prompt"].append(prompt)
            src_dict["image"].append(image)
            image_list.append( PILToTensor()(image))
            src_dict["model"].append(model)
            score=aesthetic_fn(image,{},{})[0].numpy()[0]
            src_dict["score"].append(score)
            src_dict["name"].append(name)
            #table_data.append([wandb.Image(image), model, prompt, score])
            total_score+=score
            score_list.append(score)
            try:
                slurm_job_id=os.environ["SLURM_JOB_ID"]
                with open(f"slurm/out/{slurm_job_id}.out","a+") as file:
                    print("\n",prompt,score,file=file)
            except:
                print(prompt,score)
        score_std=np.std(score_list)
        result_dict[model]["std"]=score_std
        result_dict[model]["mean"]=total_score/len(score_list)
        image_tensor=torch.stack(image_list)
        inception.update(image_tensor)
        inception_mean, inception_std=inception.compute()
        print("inception mean", inception_mean, "inception std", inception_std)
        result_dict[model]["inception_mean"]=float(inception_mean)
        result_dict[model]["inception_src"]=float(inception_std)
        try:
            slurm_job_id=os.environ["SLURM_JOB_ID"]
            with open(f"slurm/out/{slurm_job_id}.out","a+") as file:
                print(f"\ntotal score {model} {total_score} std {score_std}",file=file)
        except:
            print(f"total score {model} {total_score} std {score_std}")

    #run = wandb.init(project="creative_clip")
    #table=wandb.Table(columns=columns,data=table_data)
    #run.log({"table":table})
    Dataset.from_dict(src_dict).push_to_hub(args.hf_dir)
    model_card_content=f"created a total of {len(score_list)} images \n"
    for model,metric_dict in result_dict.items():
        model_card_content+="\n"+model
        for m_name,m_value in metric_dict.items():
            m_value=round(m_value,3)
            model_card_content+=f" {m_name}: {m_value} "
    with open("tmp.md","w+") as file:
        file.write(model_card_content)
    upload_file(path_or_fileobj="tmp.md", 
                path_in_repo="README.md",
                repo_id=args.hf_dir,
                repo_type="dataset")
    try:
        slurm_job_id=os.environ["SLURM_JOB_ID"]
        with open(f"slurm/out/{slurm_job_id}.out","a+") as file:
            print(result_dict,file=file)
            print("all done :)))",file=file)
    except:
        print(result_dict)
        print("all done :)))")