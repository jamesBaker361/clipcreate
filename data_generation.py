import os
import sys
import re
from PIL import Image
from datasets import Dataset,load_dataset
from lavis.models import load_model_and_preprocess
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

IMG_DIR='/scratch/jlb638/wikiartimages/images/'

IMAGE_STR="image"
TEXT_STR="text"
STYLE_STR="style"
NAME_STR="name"

SPLIT_FRACTION=0.1

src_dict={
    IMAGE_STR:[],
    TEXT_STR:[],
    STYLE_STR:[],
    NAME_STR:[]
}

def make_ds(limit=5):
    for style in os.listdir(IMG_DIR):
        sub_dir=IMG_DIR+style
        if os.path.isdir(sub_dir):
            print(style)
            for img_name in os.listdir(sub_dir):
                if limit<0:
                    break
                if img_name.endswith("jpg"):
                    raw_image=Image.open(sub_dir+"/"+img_name).convert("RGB")
                    src_dict[IMAGE_STR].append(raw_image)
                    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    # generate caption
                    caption=blip_model.generate({"image": image})[0]
                    src_dict[TEXT_STR].append(caption)
                    src_dict[STYLE_STR].append(style)
                    src_dict[NAME_STR].append(re.sub('.jpg!Blog.jpg',"",img_name))
                    limit-=1

    return Dataset.from_dict(src_dict).train_test_split(SPLIT_FRACTION)

if __name__=='__main__':
    name_list=["jlbaker361/wikiart20", "jlbaker361/wikiart"]
    for name,limit in zip (name_list, [20,1000000]):
        make_ds(limit=limit).push_to_hub(name)
    for name in name_list:
        load_dataset(name)