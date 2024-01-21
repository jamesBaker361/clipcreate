import os
import sys
import argparse
from static_globals import *

def get_prompt_fn(dataset_name,split):
    hf_dataset=load_dataset(dataset_name,split=split)
    prompt_list=[t for t in hf_dataset["text"]]

    def _fn():
        return random.choice(prompt_list),{}
    
    return _fn
def get_image_sample_hook(image_dir):
    os.makedirs(image_dir, exist_ok=True)
    def _fn(prompt_image_data, global_step, tracker):
        for row in prompt_image_data:
            images=row[0]
            prompts=row[1]
            for img,pmpt in zip(images, prompts):
                path=image_dir+pmpt.replace(" ", "_")+str(global_step)+".png"
                print("saving at ",path)
                pil_img=to_pil_image(img)
                pil_img.save(path)
    return _fn

parser = argparse.ArgumentParser(description="ddpo training")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default="runwayml/stable-diffusion-v1-5",
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="jlbaker361/wikiart-balanced100",
    help="The name of the Dataset (from the HuggingFace hub) to train on"
)
parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/jlb638/sd-ddpo",
        help="The output directory where the model predictions and checkpoints will be written.",
)

parser.add_argument("--image_dir",type=str,default=None)
parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
parser.add_argument(
    "--hub_model_id",
    type=str,
    default=None,
    help="The name of the repository on hf to write to",
)
parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--sample_num_steps",type=int,default=10, help="Number of sampler inference steps per image")
parser.add_argument("--sample_batch_size",type=int,default=4, help="batch size for all gpus, must be >= train_batch_size")
parser.add_argument("--train_gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--style_list",nargs="*",help="styles to be used")
parser.add_argument("--sample_num_batches_per_epoch",type=int,default=8)
parser.add_argument("--use_lora",type=bool,default=True)
parser.add_argument("--mixed_precision",type=str, default="no",help="precision, one of no, fp16, bf16")
parser.add_argument("--cache_dir",type=str, default=None)
parser.add_argument("--resume_from",type=str,default=None)
parser.add_argument("--reward_function",default="clip",type=str,help="reward function: resnet dcgan or clip")

parser.add_argument("--dcgan_repo_id",type=str,help="hf repo whre dcgan discriminator weights are",default="jlbaker361/dcgan-wikiart")
parser.add_argument("--disc_init_dim",type=int,default=32, help="initial layer # of channels in discriminator")
parser.add_argument("--disc_final_dim",type=int, default=512, help="final layer # of channels in discriminator")
parser.add_argument("--resize_dim",type=int,default=512,help="dim to resize images to before cropping (for dcgan)")


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
    #os.symlink("~/.cache/huggingface/", cache_dir)
    from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
    from huggingface_hub.utils import EntryNotFoundError
    from torchvision.transforms.functional import to_pil_image
    import torch
    import time
    from creative_loss import clip_scorer_ddpo, elgammal_resnet_scorer_ddpo, elgammal_dcgan_scorer_ddpo
    from huggingface_hub import create_repo, upload_folder, ModelCard
    from datasets import load_dataset
    import random

    #sanity check to make sure we are logged in to huggingface
    os.makedirs("test",exist_ok=True)
    test_repo_id=create_repo("jlbaker361/test", exist_ok=True).repo_id
    upload_folder(repo_id=test_repo_id, folder_path="test")


    style_list=args.style_list
    if style_list is None or len(style_list)<2:
        style_list=WIKIART_STYLES
    if args.reward_function == "clip":
        reward_fn=clip_scorer_ddpo(style_list)
    elif args.reward_function == "resnet":
        reward_fn=elgammal_resnet_scorer_ddpo(style_list,224)
    elif args.reward_function=="dcgan":
        reward_fn=elgammal_dcgan_scorer_ddpo(style_list,512, args.resize_dim, args.disc_init_dim, args.disc_final_dim, args.dcgan_repo_id)
    else:
        raise Exception("unknown reward function; should be one of clip or resnet or dcgan")
    prompt_fn=get_prompt_fn(args.dataset_name, "train")


    project_kwargs={
            "project_dir":args.output_dir,
            'automatic_checkpoint_naming':True
        }
    try:
        pipeline = DefaultDDPOStableDiffusionPipeline(
            args.pretrained_model_name_or_path,  use_lora=args.use_lora
        )
    except (EntryNotFoundError,ValueError) as error:
        print("EntryNotFoundError or ValueError using pipeline.sd_pipeline.load_lora_weights")
        pipeline=DefaultDDPOStableDiffusionPipeline("runwayml/stable-diffusion-v1-5")
        pipeline.sd_pipeline.load_lora_weights(args.pretrained_model_name_or_path,weight_name="pytorch_lora_weights.safetensors")

    resume_from=None
    if args.resume_from:
        resume_from = os.path.normpath(os.path.expanduser(args.resume_from))
        if os.path.exists(resume_from):
            checkpoints = list(
                filter(
                    lambda x: "checkpoint_" in x,
                    os.listdir(resume_from),
                )
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {resume_from}")
            checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
            resume_from = os.path.join(
                resume_from,
                f"checkpoint_{checkpoint_numbers[-1]}",
            )

            project_kwargs["iteration"] = checkpoint_numbers[-1] + 1

    config=DDPOConfig(
        num_epochs=args.num_epochs,
        train_gradient_accumulation_steps=args.train_gradient_accumulation_steps,
        sample_num_steps=args.sample_num_steps,
        sample_batch_size=args.sample_batch_size,
        train_batch_size=args.train_batch_size,
        sample_num_batches_per_epoch=args.sample_num_batches_per_epoch,
        mixed_precision=args.mixed_precision,
        #resume_from=args.resume_from,
        accelerator_kwargs={
            "project_dir":args.output_dir
        },
        project_kwargs=project_kwargs
    )

    if args.image_dir==None:
        args.image_dir="images"
        os.makedirs(args.image_dir, exist_ok=True)
    image_samples_hook=get_image_sample_hook(args.image_dir)
    trainer = DDPOTrainer(
            config,
            reward_fn,
            prompt_fn,
            pipeline
    )
    if resume_from:
        print(f"Resuming from {resume_from}")
        try:
            pipeline.sd_pipeline.load_lora_weights(resume_from,weight_name="pytorch_lora_weights.safetensors")
            trainer.first_epoch=int(resume_from.split("_")[-1]) + 1
        except:
            print(f"could not resume from {resume_from}")

    start=time.time()
    torch.cuda.memory._record_memory_history()
    try:
        trainer.train()
    except Exception as exc:
        print(exc)
        torch.cuda.memory._dump_snapshot("failure.pickle")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.max_memory_allocated: %fGB"%(torch.cuda.max_memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print(torch.cuda.memory_summary())
        print("torch.cuda.list_gpu_processes()",torch.cuda.list_gpu_processes())
        raise exc
    torch.cuda.memory._dump_snapshot("success.pickle")
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful training :) time elapsed: {seconds} seconds = {hours} hours")

    trainer._save_pretrained(args.output_dir)
    repo_id = create_repo(repo_id=args.hub_model_id, exist_ok=True).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path=args.output_dir,
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
    )
    model_card_content=f"""
    # DDPO trained model
    num_epochs={args.num_epochs} \n
    train_gradient_accumulation_steps={args.train_gradient_accumulation_steps} \n
    sample_num_steps={args.sample_num_steps} \n
    sample_batch_size={args.sample_batch_size} \n 
    train_batch_size={args.train_batch_size} \n
    sample_num_batches_per_epoch={args.sample_num_batches_per_epoch} \n

    """
    card=ModelCard(model_card_content)
    card.push_to_hub(repo_id)
    print("successful saving :)")
