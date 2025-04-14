import diffusers
import huggingface_hub
from PIL import Image
import os
from image_diversity import ClipMetrics

clip_metrics = ClipMetrics()


data=[f"F{x}" for x in range(3)]+[f"M{x}" for x in range(3)]

def calculate_mode_collapse(image_list:list[Image.Image]):
    dir_name="temp"
    os.makedirs(dir_name,exist_ok=True)
    for i,image in enumerate(image_list):
        image.save(f"{dir_name}/{i}.png")
    tce = clip_metrics.tce(dir_name)