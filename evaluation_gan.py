import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir

from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
import random
import argparse
from static_globals import *

random.seed(1234)

parser = argparse.ArgumentParser(description="evaluation")

def evaluate(args):
    pass


if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    evaluate(args)