import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
import torch

try:
    pipeline = DiffusionPipeline.from_pretrained("jlbaker361/sd-wikiart20")
    prompt = "cat"
    image = pipeline(prompt).images[0]
    image.save("DiffusionPipeline_image.png")
except Exception as exc:
    print(exc)
    print("failed for normal pipeline")


try:
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.load_lora_weights("jlbaker361/sd-wikiart20", weight_name="pytorch_lora_weights.safetensors", adapter_name="wikiart20")
    prompt = "cat"
    image = pipeline(prompt).images[0]
    image.save("lora_DiffusionPipeline_image.png")
except Exception as exc:
    print(exc)
    print("failed for lora pipeline")


try:
    pipeline = StableDiffusionPipeline.from_pretrained("jlbaker361/sd-wikiart20")
    prompt = "cat"
    image = pipeline(prompt).images[0]
    image.save("stableDiffusionPipeline_image.png")
except Exception as exc:
    print(exc)
    print("failed for stable diffusion pipeline")


try:
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.load_lora_weights("jlbaker361/sd-wikiart20", weight_name="pytorch_lora_weights.safetensors", adapter_name="wikiart20")
    prompt = "cat"
    image = pipeline(prompt).images[0]
    image.save("lora_stableDiffusionPipeline_image.png")
except Exception as exc:
    print(exc)
    print("failed for stable lora pipeline")

try:
    pipeline = DefaultDDPOStableDiffusionPipeline("jlbaker361/sd-wikiart20", use_lora=False)
    prompt = "cat"
    image = pipeline(prompt).images[0]
    image.save("ddpo_DiffusionPipeline_image.png")
except Exception as exc:
    print(exc)
    print("failed for ddpo normal pipeline")


try:
    pipeline = DefaultDDPOStableDiffusionPipeline("jlbaker361/sd-wikiart20", use_lora=True)
    prompt = "cat"
    image = pipeline(prompt).images[0]
    image.save("ddpo_lora_DiffusionPipeline_image.png")
except Exception as exc:
    print(exc)
    print("failed for DefaultDDPOStableDiffusionPipeline(jlbaker361/sd-wikiart20, use_lora=True)")

try:
    pipeline = DefaultDDPOStableDiffusionPipeline("jlbaker361/sd-wikiart20-lora", use_lora=True)
    prompt = "cat"
    image = pipeline(prompt).images[0]
    image.save("ddpo_lora_lora_DiffusionPipeline_image.png")
except Exception as exc:
    print(exc)
    print("failed for DefaultDDPOStableDiffusionPipeline(jlbaker361/sd-wikiart20-lora, use_lora=True)")


try:
    pipeline = DefaultDDPOStableDiffusionPipeline("runwayml/stable-diffusion-v1-5", use_lora=True)
    pipeline.load_lora_weights("jlbaker361/sd-wikiart20", weight_name="pytorch_lora_weights.safetensors", adapter_name="wikiart20")
    prompt = "cat"
    image = pipeline(prompt).images[0]
    image.save("ddpo_lora_lora_2_DiffusionPipeline_image.png")
except Exception as exc:
    print(exc)
    print("failed for DefaultDDPOStableDiffusionPipeline(runwayml/stable-diffusion-v1-5, use_lora=True)")