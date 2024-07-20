#use this to generate the synthetic dataset
import random
from diffusers import DiffusionPipeline
import torch
from datasets import Dataset

adjective_list=["a","a cool","a happy","a sad"]
subject_list=["man","woman","person"]
verb_list=["walking", "reading", "eating", "running", "posing"]
location_list=["in a park", "at home", "at work", "in the forest", "in a city"]
style_list=["graffiti", "pop art", "anime", "comic book", "picture book"]
negative="ugly, low resolution, blurry, horror"

limit=700
target_dataset="jlbaker361/synthetic-data"

src_dict={
    "image":[],
    "style":[],
    "text":[]
}

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
for step in range(limit):
    list_of_lists=[
        adjective_list,
        subject_list,
        verb_list,
        location_list,
    ]
    text=" ".join([random.choice(_list) for _list in list_of_lists])
    src_dict["text"].append(text)
    style=random.choice(style_list)
    text+=f", {style} style"
    src_dict["style"].append(style.replace(" ","_"))
    image = base(
    prompt=text,
        num_inference_steps=40,
        denoising_end=0.8,
        output_type="latent",
        negative_prompt=negative
    ).images
    image = refiner(
        prompt=text,
        num_inference_steps=40,
        denoising_start=0.8,
        image=image,
        negative_prompt=negative
    ).images[0]
    src_dict["image"].append(image)
    if step % 100==0:
        Dataset.from_dict(src_dict).push_to_hub(target_dataset)
        print("pushed at step ",step)
print("all done :)))")