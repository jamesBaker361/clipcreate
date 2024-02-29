
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
from safetensors import safe_open
from safetensors.torch import save_file
from peft import get_peft_model_state_dict
from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline

def prompt_fn():
    return "a painting", {}

clip_reward_fn=clip_scorer_ddpo(["baroque", "cubism"])
aesthetic_reward_fn=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)

def save_lora_weights(pipeline:BetterDefaultDDPOStableDiffusionPipeline,output_dir:str):
    state_dict=get_peft_model_state_dict(pipeline.sd_pipeline.unet, unwrap_compiled=True)
    weight_path=os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    print("saving to ",weight_path)
    save_file(state_dict, weight_path, metadata={"format": "pt"})

def load_lora_weights(pipeline:BetterDefaultDDPOStableDiffusionPipeline,output_dir:str):
    #pipeline.get_trainable_layers()
    path=os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    print("loading from ",path)
    state_dict={}
    count=0
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key]=f.get_tensor(key)
    state_dict={
        k.replace("weight","default.weight"):v for k,v in state_dict.items()
    }
    pipeline.sd_pipeline.unet.load_state_dict(state_dict,strict=False)

def train_test(mixed_precision="no", 
               reward_fn=clip_reward_fn,
               src_hub_model_id="runwayml/stable-diffusion-v1-5",
               hub_model_id="jlbaker361/sd-ddpo-wikiart20",
               save_directory="/scratch/jlb638/sd-ddpo-test"):

    weight_name="down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight"
    config=DDPOConfig(
        num_epochs=1,
                train_gradient_accumulation_steps=1,
                sample_num_steps=2,
                sample_batch_size=1,
                train_batch_size=1,
                sample_num_batches_per_epoch=1,
                mixed_precision=mixed_precision,
                train_learning_rate=0.03
    )
    pipeline = BetterDefaultDDPOStableDiffusionPipeline(
            src_hub_model_id,  use_lora=True
        )
    print("before training:")
    print(pipeline.sd_pipeline.unet.get_parameter('down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight').detach().numpy()[0][:10])

    trainer = DDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline
        )
    trainer.train()
    print("successful training boys :)))")
    print("after training:")
    print(pipeline.sd_pipeline.unet.get_parameter('down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight').detach().numpy()[0][:10])
    save_lora_weights(pipeline, save_directory)
    load_lora_weights(pipeline, save_directory)
    pipeline = BetterDefaultDDPOStableDiffusionPipeline(
            src_hub_model_id,  use_lora=True
        )
    print("before loading:")
    print(pipeline.sd_pipeline.unet.get_parameter('down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight').detach().numpy()[0][:10])
    
    load_lora_weights(pipeline, save_directory)
    print("after loading")
    print(pipeline.sd_pipeline.unet.get_parameter('down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight').detach().numpy()[0][:10])
    exit()
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
    

    #both of these ways to load work btw
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