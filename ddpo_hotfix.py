import os
import sys
import argparse
from static_globals import *

parser = argparse.ArgumentParser(description="ddpo training")

if __name__=='__main__':
    args = parser.parse_args()
    print(args)
    if args.cache_dir is not None:
        os.makedirs(args.cache_dir,exist_ok=True)
        os.environ["TRANSFORMERS_CACHE"]=args.cache_dir
        os.environ["HF_HOME"]=args.cache_dir
        os.environ["HF_HUB_CACHE"]=args.cache_dir
        torch_cache_dir=args.cache_dir+"/torch"
        os.makedirs(torch_cache_dir,exist_ok=True)
        import torch
        torch.hub.set_dir(torch_cache_dir)
    from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
    from huggingface_hub.utils import EntryNotFoundError
    from torchvision.transforms.functional import to_pil_image
    import torch
    import time
    from creative_loss import clip_scorer_ddpo, elgammal_scorer_ddpo
    from huggingface_hub import create_repo, upload_folder, ModelCard
    from datasets import load_dataset
    import random

    #sanity check to make sure we are logged in to huggingface
    os.makedirs("test",exist_ok=True)
    test_repo_id=create_repo("jlbaker361/test", exist_ok=True).repo_id
    upload_folder(repo_id=test_repo_id, folder_path="test")

    pipeline=DefaultDDPOStableDiffusionPipeline("runwayml/stable-diffusion-v1-5")
    resume_from="~/clipcreat/vanilla-ddpo25/checkpoints/checkpoint_13"
    pipeline.sd_pipeline.load_lora_weights(resume_from,weight_name="pytorch_lora_weights.safetensors")