import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from generator_src import Generator
import time
import torch
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToPILImage
from datasets import load_dataset, Dataset

parser=argparse.ArgumentParser()

parser.add_argument("--limit",type=int,default=100)
parser.add_argument("--project_name",type=str,default="evaluation-dcgan")
parser.add_argument("--output_hf_dataset",type=str,default="evaluation-dcgan")
parser.add_argument("--hub_model_id",type=str,default="jlbaker361/can-512-0.1-full")
parser.add_argument("--mixed_precision",type=str,default="no")



def main(args):
    gen=Generator(100,512,False)
    gen_weight_path=hf_hub_download(args.hub_model_id, filename="gen-weights.pickle")
    if torch.cuda.is_available():
        gen_state_dict=torch.load(gen_weight_path)
    else:
        gen_state_dict=torch.load(gen_weight_path,map_location=torch.device('cpu'))
    gen.load_state_dict(gen_state_dict)


    src_dict={
        "image":[],
        "prompt":[],
        "index":[]
    }
    torch.manual_seed(0)
    try:
        torch.cuda.manual_seed(0)
    except:
        pass
    for i in range(args.limit):
        prompt="art"
        noise=torch.randn(1,100,1,1)
        image =ToPILImage()(gen(noise,None)[0])
        
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