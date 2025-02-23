import os
import argparse
from experiment_helpers.gpu_details import print_details
from creative_loss import image_reward_scorer, clip_scorer_ddpo, clip_prompt_alignment,elgammal_dcgan_scorer_ddpo
from aesthetic_reward import aesthetic_scorer,hf_hub_aesthetic_model_id,hf_hub_aesthetic_model_filename
from accelerate import Accelerator
from datasets import load_dataset
import time
from experiment_helpers.measuring import get_vit_embeddings,get_metric_dict, cos_sim
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel
from experiment_helpers.static_globals import *
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.aesthetic_reward import AestheticScorer
import ImageReward as image_reward
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import wandb

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--dataset_list",nargs="*")
parser.add_argument("--prompt_set",type=str,default="all")
parser.add_argument("--name",type=str,default="name")
parser.add_argument("--dcgan_repo_id",type=str,help="hf repo whre dcgan discriminator weights are",default="jlbaker361/dcgan-vanilla")
parser.add_argument("--limit",type=int,default=1000)
parser.add_argument("--src_dataset",type=str,default="jlbaker361/wikiart")


prompt_set_dict={
    "mediums":["painting","art","drawing"],
    "subjects":["person","man","woman"],
    "all":["painting","art","drawing","person","man","woman"]
}

@torch.no_grad()
def main(args):
    vit_processor=ViTImageProcessor.from_pretrained("facebook/dino-vits16")
    vit_model=BetterViTModel.from_pretrained("facebook/dino-vits16").eval()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    vit_model.to(accelerator.device)
    vit_model=accelerator.prepare(vit_model)
    clip_model.to(accelerator.device)
    clip_model=accelerator.prepare(clip_model)
    aesthetic_scorer=AestheticScorer()
    aesthetic_scorer.to(accelerator.device)
    aesthetic_scorer=accelerator.prepare(aesthetic_scorer)
    #ir_scorer=image_reward_scorer(accelerator)
    src_dataset=load_dataset(args.src_dataset,split="train")
    style_set=set()
    for row in src_dataset:
        style_set.add(row["style"])
    style_list=list(style_set)
    create_scorer=clip_scorer_ddpo(style_list)
    n_classes=27
    if args.name=="wikiart-mediums" or args.name=="wikiart-subjects":
        n_classes=10
    elif args.name=="synthetic-data":
        n_classes=5
    disc_scorer=elgammal_dcgan_scorer_ddpo([_ for _ in range(n_classes)],512,512,32,512,args.dcgan_repo_id,device=accelerator.device)
    #a_scorer=aesthetic_scorer(hf_hub_aesthetic_model_id, hf_hub_aesthetic_model_filename)

    dataset_grid=[[row for row in load_dataset(f"{data}",split="train")] for data in args.dataset_list]
    style_difference_matrix=[[[] for _ in args.dataset_list] for __ in args.dataset_list]
    content_difference_matrix=[[[] for _ in args.dataset_list] for __ in args.dataset_list]
    DISC_CREATIVITY="disc_creativity"
    CLIP_CREATIVITY="clip_creativity"
    ir_model=image_reward.load("/scratch/jlb638/reward-blob",med_config="/scratch/jlb638/ImageReward/med_config.json",device=accelerator.device)
    score_matrix=[{
        AESTHETIC_SCORE:[],
        IMAGE_REWARD:[],
        #PROMPT_SIMILARITY:[],
        DISC_CREATIVITY:[],
        CLIP_CREATIVITY:[]
    } for _ in args.dataset_list]
    n_images=len(dataset_grid[0])
    for y in range(n_images):
        if y>=args.limit:
            break

        image_list=[]
        vit_style_list=[]
        vit_content_list=[]
        for x in range(len(args.dataset_list)):
            try:
                image=dataset_grid[x][y]["image"]
            except:
                print(f"x {x} len {len(args.dataset_list)}")
                print(f"y {y} len {len(dataset_grid[x])}")
                raise Exception("listt index shit???")
            new_image=Image.new('RGB', image.size)
            new_image.paste(image)
            image=new_image
            prompt=dataset_grid[x][y]["prompt"]
            '''clip_inputs=clip_processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
            clip_inputs["input_ids"]=clip_inputs["input_ids"].to(clip_model.device)
            clip_inputs["pixel_values"]=clip_inputs["pixel_values"].to(clip_model.device)
            clip_inputs["attention_mask"]=clip_inputs["attention_mask"].to(clip_model.device)
            try:
                clip_inputs["position_ids"]= clip_inputs["position_ids"].to(clip_model.device)
            except:
                pass
            clip_outputs = clip_model(**clip_inputs)
            image_embeds=clip_outputs.image_embeds.detach().cpu().numpy()[0]
            text_embeds=clip_outputs.text_embeds.cpu().detach().numpy()[0]
            prompt_distance=cos_sim(image_embeds, text_embeds)'''
            #score_matrix[x][PROMPT_SIMILARITY].append(prompt_distance)

            score_matrix[x][AESTHETIC_SCORE].append(aesthetic_scorer(image).cpu().numpy()[0])
            score_matrix[x][IMAGE_REWARD].append(ir_model.score(prompt,image))
            _, style_emb,content_emb =get_vit_embeddings(vit_processor, vit_model,[image])
            image_list.append(image)
            vit_style_list.append(style_emb[0])
            vit_content_list.append(content_emb[0])
            clip_create_score=create_scorer([image],None,None)[0][0]
            disc_create_score=disc_scorer([image],None,None)[0][0]
            score_matrix[x][DISC_CREATIVITY].append(disc_create_score)
            score_matrix[x][CLIP_CREATIVITY].append(clip_create_score)
        for i in range(len(args.dataset_list)):
            style_difference_matrix[i][i].append(1.0)
            content_difference_matrix[i][i].append(1.0)
            for j in range(i+1,len(args.dataset_list)):
                cont_sim=cos_sim(vit_content_list[i],vit_content_list[j])
                style_sim=cos_sim(vit_style_list[i], vit_style_list[j])
                style_difference_matrix[i][j].append(style_sim)
                style_difference_matrix[j][i].append(style_sim)
                content_difference_matrix[i][j].append(cont_sim)
                content_difference_matrix[j][i].append(cont_sim)
    
    for i,name in enumerate(args.dataset_list):
        print(f"\\(M_{i}\\)", " & ",end=" ")
        score_dict=score_matrix[i]
        for k,v in score_dict.items():
            try:
                v=[_v.cpu() for _v in v]
            except:
                pass
            print(np.round(np.mean(v),2),f" (",np.round(np.std(v),2),") & ",end=" ")
            metric_root=f"M{i}_{k}"
            accelerator.log({
                metric_root+"_mean":np.mean(v),
                metric_root+"_std":np.std(v)
            })
        print("\\\\")
    style_avg=[[np.mean(scores) for scores in score_list] for score_list in style_difference_matrix]
    for row in style_avg:
        print(" & ".join([str(np.round(r,2)) for r in row]))
    content_avg=[[np.mean(scores) for scores in score_list] for score_list in content_difference_matrix]
    print("content")
    for row in content_avg:
        print(" & ".join([str(np.round(r,2)) for r in row]))
    #labels=[f"M{x}" for x in range(len(args.dataset_list))]
    labels=["disc\n-full","clip\n-full","kmeans\n-full","disc\n-med","clip\n-med","kmeans\n-med","utility\n-30","utility\n-10","basic\n-30","basic\n-10"][:len(args.dataset_list)]

    accelerator.log({
        "style_table":wandb.Table(columns=[f"M{x}" for x in range(len(args.dataset_list))],data=style_avg),
        "content_table":wandb.Table(columns=[f"M{x}" for x in range(len(args.dataset_list))],data=content_avg)
    })

    for metric in [IMAGE_REWARD,AESTHETIC_SCORE]:
        reward_data=[
            score_matrix[x][metric] for x in range(len(args.dataset_list))
        ]

        plt.boxplot(reward_data)

        # Set the title and labels
        #plt.title('Boxplot of 7 Lists of Values')
        title={
            "wikiart-mediums": "WikiArt Mediums",
            "wikiart": "WikiArt Full",
            "wikiart-subjects": "WikiArt Subjects",
            "everyone":"",
            "shitty":"Models Trained Without Creativity",
            "only_trained":"",
            "november":""
        }[args.name]
        title+={
            IMAGE_REWARD:" Image Reward",
            AESTHETIC_SCORE: " Aesthetic Score"
        }[metric]
        #plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title(title)

        # Set the x-axis tick labels
        plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)

        plt.savefig(f"boxes/{args.name}_{metric}.png")
        plt.show()
        plt.clf()


            


    

if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")