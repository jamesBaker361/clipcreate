import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")

import argparse
parser = argparse.ArgumentParser(description="image comparisons")

def main(args):
    pass

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    main(args)