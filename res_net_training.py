import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from res_net_src import ResNet, HFDataset
from accelerate import Accelerator
import argparse
parser = argparse.ArgumentParser(description="ddpo training")
parser.add_argument("--epochs", type=int,default=100)
parser.add_argument("--dataset_name",type=str,default="jlbaker361/wikiart")
parser.add_argument("--pretrained_version",type=str, default="resnet18")
def training_loop(epochs:int, dataset_name:str, pretrained_version:str):
    pass

if __name__=="__main__":
    args=parser.parse_args()
    training_loop(args.epochs, args.dataset_name,args.pretrained_version)