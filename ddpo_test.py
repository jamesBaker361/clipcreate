
import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
#os.symlink("~/.cache/huggingface/", cache_dir)
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration,BlipModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from torch.nn import Softmax
import torch
from creative_loss import clip_scorer_ddpo
from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
from huggingface_hub import create_repo, upload_folder

def prompt_fn():
    return "a painting", {}

clip_reward_fn=clip_scorer_ddpo(["baroque", "cubism"])
aesthetic_reward_fn=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)

def train_test(mixed_precision="no", 
               reward_fn=clip_reward_fn,
               src_hub_model_id="jlbaker361/sd-wikiart20",
               hub_model_id="jlbaker361/sd-ddpo-wikiart20",
               save_directory="/scratch/jlb638/sd-ddpo-test"):

    config=DDPOConfig(
        num_epochs=1,
                train_gradient_accumulation_steps=1,
                sample_num_steps=2,
                sample_batch_size=2,
                train_batch_size=1,
                sample_num_batches_per_epoch=2,
                mixed_precision=mixed_precision
    )

    pipeline = DefaultDDPOStableDiffusionPipeline(
            src_hub_model_id,  use_lora=True
        )

    trainer = DDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline
        )
    trainer.train()
    print("successful training boys :)))")
    trainer._save_pretrained(save_directory)
    repo_id = create_repo(repo_id=hub_model_id, exist_ok=True).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path=save_directory,
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
    )
    print("successful saving boys :)))")
    try:
        pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipeline.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors", adapter_name="wikiart20")
        prompt = "painting"
        image = pipeline(prompt).images[0]
        image.save("ddpo_test_lora.png")
        print("success for pipeline.load_lora_weights(repo_id")
    except Exception as exc:
        print(exc)
        print("failure for pipeline.load_lora_weights(repo_id")
    try:
        pipeline = DefaultDDPOStableDiffusionPipeline(repo_id, use_lora=True)
        prompt = "painting"
        image = pipeline(prompt).images[0]
        image.save("ddpo_test.png")
        print("success for DefaultDDPOStableDiffusionPipeline(repo_id")
    except Exception as exc:
        print(exc)
        print("failed for DefaultDDPOStableDiffusionPipeline(repo_id")
    

if __name__=='__main__':
    print(sys.argv)
    if "aesthetic" in sys.argv:
        train_test(reward_fn=aesthetic_reward_fn)
    else:
        train_test()
    '''bad_mp_list=[]
    for mp in sys.argv[1:]:
        try:
            train_test(mp)
        except RuntimeError as run:
            print(run)
            print("runtime error for mp =", mp)
            bad_mp_list.append(mp)
    print("doesnt work for :", bad_mp_list)'''