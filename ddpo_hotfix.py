import os
import sys
import argparse
from static_globals import *

parser = argparse.ArgumentParser(description="ddpo training")

if __name__=='__main__':
    args = parser.parse_args()
    print(args)
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

    pipeline=DefaultDDPOStableDiffusionPipeline("runwayml/stable-diffusion-v1-5",use_lora=True)
    resume_from="vanilla-ddpo25/checkpoints/checkpoint_13"
    print("loading???")
    pipeline.sd_pipeline.load_lora_weights(resume_from,weight_name="pytorch_lora_weights.safetensors")
    print("loaded :)")

    config=DDPOConfig(
        num_epochs=0
    )
    def get_prompt_fn(dataset_name,split):
        hf_dataset=load_dataset(dataset_name,split=split)
        prompt_list=[t for t in hf_dataset["text"]]

        def _fn():
            return random.choice(prompt_list),{}
        return _fn

    reward_fn=clip_scorer_ddpo(["dw about it"])
    prompt_fn=get_prompt_fn("jlbaker361/wikiart-balanced250", "train")

    trainer = DDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline
    )

    trainer._save_pretrained("vanilla-ddpo25/")
    repo_id = create_repo(repo_id="jlbaker361/vanilla-ddpo25", exist_ok=True).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path="vanilla-ddpo25",
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
    )
    model_card_content=f"""
    # DDPO trained model
    pls train for one more epoch

    """
    card=ModelCard(model_card_content)
    card.push_to_hub(repo_id)
    print("successful saving :)")