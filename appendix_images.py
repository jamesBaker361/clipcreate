import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from datasets import Dataset,load_dataset
from huggingface_hub import hf_hub_download
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
import random
import matplotlib.pyplot as plt
from matplotlib.table import table
from accelerate import Accelerator
import numpy as np
from generator_src import Generator
from sentence_transformers import SentenceTransformer
from torchvision.transforms import ToPILImage

import argparse
parser = argparse.ArgumentParser(description="image comparisons")
parser.add_argument("--conditional",type=bool, default=False)
parser.add_argument("--can_model_list",nargs="*",help="list of CAN models to test")
parser.add_argument("--ddpo_model_list",nargs="*",help="list of ddpo models to help")
parser.add_argument("--limit",type=int,default=5,help="n images")
parser.add_argument("--dataset",type=str,help="source of prompts",default="jlbaker361/wikiart-balanced1000")
parser.add_argument("--gen_z_dim",type=int,default=100,help="dim latent noise for generator")
parser.add_argument("--image_dim", type=int,default=512)
parser.add_argument("--file_path",type=str,default="table.png",help="file to save table")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--lora_scale", type=float,default=0.9, help="lora scale [0.0-1.0]")


# Function to create a table with prompts and images
def create_row(ax, row_data,conditional):
    prompt = row_data["prompt"]
    images = row_data["images"]

    offset=0
    if conditional:
        # Plot prompt
        ax[0].text(0.5, 0.5, prompt, va='center', ha='center', fontsize=10, color='black')
        ax[0].axis('off')
        offset=1

    # Plot images
    for i, image in enumerate(images):
        ax[i + offset].imshow(image, cmap='gray')  # Adjust the colormap as needed
        ax[i + offset].axis('off')

'''data = [
    {"prompt": "Prompt 1", "images": [np.random.rand(10, 10) for _ in range(4)]},
    {"prompt": "Prompt 2", "images": [np.random.rand(10, 10) for _ in range(4)]},
    # Add more rows as needed
]'''
def save_table(data,conditional,file_path):
    # Create the plot
    rows = len(data)
    cols = len(data[0]["images"])  # 1 column for prompt and 4 columns for images
    if conditional:
        cols+=1

    fig, ax = plt.subplots(rows, cols, figsize=(10, 2 * rows))

    for i, row_data in enumerate(data):
        create_row(ax[i], row_data,conditional)

    plt.tight_layout()
    plt.show()

    plt.savefig(file_path)
    print(f"saved at {file_path}")

def main(args):
    if args.can_model_list is None:
        args.can_model_list=[]
    if args.ddpo_model_list is None:
        args.ddpo_model_list=[]

    accel=Accelerator()

    test_dataset=load_dataset(args.dataset, split="test")
    prompt_list=[t for t in test_dataset["text"]]
    random.shuffle(prompt_list)
    prompt_list=prompt_list[:args.limit]
    print(args.can_model_list+args.ddpo_model_list)
    can_model_dict={}
    ddpo_model_dict={}
    for can_model in args.can_model_list:
        gen=Generator(args.gen_z_dim, args.image_dim,args.conditional)
        weights_location=hf_hub_download(can_model, filename="gen-weights.pickle")
        if torch.cuda.is_available():
            state_dict=torch.load(weights_location)
        else:
            state_dict=torch.load(weights_location, map_location=torch.device("cpu"))
        gen.load_state_dict(state_dict)
        gen.to(accel.device)
        can_model_dict[can_model]=gen
    for ddpo_model in args.ddpo_model_list:
        pipeline=DefaultDDPOStableDiffusionPipeline("stabilityai/stable-diffusion-2-base")
        pipeline.sd_pipeline.load_lora_weights(ddpo_model,weight_name="pytorch_lora_weights.safetensors")
        #pipeline.sd_pipeline.to(accel.device)
        ddpo_model_dict[ddpo_model]=pipeline
    data=[]
    sentence_encoder=SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
    to_pil_image=ToPILImage()
    for prompt in prompt_list:
        images=[]
        text_encoding=torch.unsqueeze(torch.tensor(sentence_encoder.encode(prompt)),0)
        for model_name,gen in can_model_dict.items():
            noise=torch.randn((1,args.gen_z_dim,1,1))
            if torch.cuda.is_available():
                noise=noise.to(accel.device)
                text_encoding=text_encoding.to(accel.device)
            images.append(to_pil_image(gen(noise,text_encoding)[0]))
        for model_name,pipeline in ddpo_model_dict.items():
            if args.conditional is False:
                prompt=""
            images.append(pipeline(prompt, num_inference_steps=args.num_inference_steps,cross_attention_kwargs={"scale": args.lora_scale}).images[0])
        data.append({"prompt":prompt,"images":images})
    save_table(data,args.conditional,args.file_path)


    

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done! :)))")