import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import sys
import re
from PIL import Image
from datasets import Dataset,load_dataset
from lavis.models import load_model_and_preprocess
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#blip_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

cache_dir="/scratch/jlb638/trans_cache"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",cache_dir=cache_dir)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
qa_model=BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", cache_dir=cache_dir)
caption_model.to(device)
qa_model.to(device)

IMG_DIR='/scratch/jlb638/wikiartimages/images/'

IMAGE_STR="image"
TEXT_STR="text"
STYLE_STR="style"
NAME_STR="name"
GEN_STYLE_STR="gen_style" # the style generated by asking what the model style the thing was

GEN_QUESTION="what art style is the picture?"

SPLIT_FRACTION=0.1

src_dict={
    IMAGE_STR:[],
    TEXT_STR:[],
    STYLE_STR:[],
    NAME_STR:[],
    GEN_STYLE_STR:[]
}

def make_ds(limit=5):
    for style in os.listdir(IMG_DIR):
        sub_dir=IMG_DIR+style
        if os.path.isdir(sub_dir):
            print(style)
            for img_name in os.listdir(sub_dir):
                if img_name.endswith("jpg"):
                    raw_image=Image.open(sub_dir+"/"+img_name)
                    src_dict[IMAGE_STR].append(raw_image)
                    caption_inputs = processor(images=raw_image, return_tensors="pt",padding=True).to(device)
                    caption_ids = caption_model.generate(**caption_inputs, max_new_tokens=50)
                    caption_text = processor.batch_decode(caption_ids, skip_special_tokens=True)[0].strip()
                    src_dict[TEXT_STR].append(caption_text)

                    qa_inputs=processor(images=raw_image, text=GEN_QUESTION, return_tensors="pt").to(device)
                    qa_ids = qa_model.generate(**qa_inputs)
                    qa_text=processor.batch_decode(qa_ids, skip_special_tokens=True)[0].strip()
                    src_dict[GEN_STYLE_STR].append(qa_text)

                    src_dict[STYLE_STR].append(style)
                    src_dict[NAME_STR].append(re.sub('.jpg!Blog.jpg',"",img_name))
                    limit-=1
                    if limit<0:
                        return Dataset.from_dict(src_dict).train_test_split(SPLIT_FRACTION)

    return Dataset.from_dict(src_dict).train_test_split(SPLIT_FRACTION)

def make_ds_balanced(limit=10):
    for style in os.listdir(IMG_DIR):
        sub_dir=IMG_DIR+style
        if os.path.isdir(sub_dir):
            print(style)
            count=0
            for img_name in os.listdir(sub_dir):
                if img_name.endswith("jpg") and count<limit:
                    raw_image=Image.open(sub_dir+"/"+img_name)
                    src_dict[IMAGE_STR].append(raw_image)
                    caption_inputs = processor(images=raw_image, return_tensors="pt",padding=True).to(device)
                    caption_ids = caption_model.generate(**caption_inputs, max_new_tokens=50)
                    caption_text = processor.batch_decode(caption_ids, skip_special_tokens=True)[0].strip()
                    src_dict[TEXT_STR].append(caption_text)

                    qa_inputs=processor(images=raw_image, text=GEN_QUESTION, return_tensors="pt").to(device)
                    qa_ids = qa_model.generate(**qa_inputs)
                    qa_text=processor.batch_decode(qa_ids, skip_special_tokens=True)[0].strip()
                    src_dict[GEN_STYLE_STR].append(qa_text)

                    src_dict[STYLE_STR].append(style)
                    src_dict[NAME_STR].append(re.sub('.jpg!Blog.jpg',"",img_name))
                    count+=1

    return Dataset.from_dict(src_dict).train_test_split(SPLIT_FRACTION)