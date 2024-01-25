import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from generator_src import Generator
from discriminator_src import Discriminator,GANDataset,UtilDataset

from huggingface_hub import create_repo, upload_folder, ModelCard
from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time
from static_globals import *

parser = argparse.ArgumentParser(description="classifier training")
parser.add_argument("--epochs", type=int,default=50)
parser.add_argument("--dataset_name",type=str,default="jlbaker361/wikiart")
parser.add_argument("--batch_size",type=int, default=4)
parser.add_argument("--repo_id",type=str,default="jlbaker361/dcgan-wikiart")
parser.add_argument("--output_dir",type=str,default="/scratch/jlb638/dcgan-wikiart")
parser.add_argument("--use_clip",type=bool,default=False)
parser.add_argument("--style_lambda",type=float,default=0.1, help="coefficient on style terms")

parser.add_argument("--gen_z_dim",type=int,default=100,help="dim latent noise for generator")
parser.add_argument("--image_dim", type=int,default=512)
parser.add_argument("--disc_init_dim",type=int,default=32)
parser.add_argument("--disc_final_dim",type=int,default=512)

parser.add_argument("--style_list",nargs="+",default=WIKIART_STYLES)
parser.add_argument("--resize_dim",type=int,default=512)



def training_loop(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(args.output_dir,exist_ok=True)
    gen=Generator(args.gen_z_dim,args.image_dim)
    disc=Discriminator(args.image_dim, args.disc_init_dim,args.disc_final_dim,args.style_list)
    dataset=GANDataset(args.dataset_name,args.image_dim,args.resize_dim,args.batch_size,"train")
    
    for x,y in dataset:
        break
    print(y.size())
    print(y.size()[-1])
    n_classes=y.size()[-1]
    util_dataset=UtilDataset(args.gen_z_dim, len(dataset),n_classes)
    try:
        repo_id=create_repo(repo_id=args.repo_id, exist_ok=True).repo_id
    except:
        print("retrying creating repo")
        repo_id=create_repo(repo_id=args.repo_id, exist_ok=True).repo_id

    gen_optimizer=optim.Adam(gen.parameters())
    disc_optimizer=optim.Adam(disc.parameters())
    #scheduler=optim.lr_scheduler.LinearLR(optimizer)
    training_dataloader=DataLoader(dataset, batch_size=args.batch_size,drop_last=True)
    util_dataloader=DataLoader(util_dataset,batch_size=args.batch_size,drop_last=True)

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name="creativity")
    gen, gen_optimizer, disc, disc_optimizer, training_dataloader, util_dataloader = accelerator.prepare(gen, gen_optimizer, disc, disc_optimizer, training_dataloader,util_dataloader)

    if args.use_clip:
        print("using clip classifier")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",do_rescale=False)

        model,processor=accelerator.prepare(model,processor)

        def clip_classifier(images):
            #images=images/255
            inputs = processor(text=args.style_list, images=images, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            return logits_per_image.softmax(dim=1)

    device=accelerator.device
    print(f"acceleerate device = {device}")
    #gen.to(device)
    #disc.to(device)
    cross_entropy=torch.nn.CrossEntropyLoss()
    binary_cross_entropy = torch.nn.BCELoss()
    #uniform=torch.full((args.batch_size, n_classes), fill_value=1.0/n_classes)
    real_label_int = 1.
    fake_label_int = 0.
    for e in range(args.epochs):
        style_classification_loss_sum=0.
        fake_binary_loss_sum=0.
        real_binary_loss_sum=0.
        style_ambiguity_loss_sum=0.
        reverse_fake_binary_loss_sum=0.
        start=time.time()
        for batch,util_vectors in zip(training_dataloader,util_dataloader):
            noise,real_vector,fake_vector,uniform = util_vectors
            real_images, real_labels = batch
            real_labels=real_labels.to(torch.float64)
            #real_images, real_labels = real_images.to(device), real_labels.to(device)
            #noise= torch.randn(args.batch_size, 100, 1, 1)
            #noise.to(device)
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            real_binary,real_style=disc(real_images)
            fake_images=gen(noise)
            fake_binary,fake_style=disc(fake_images)
            reverse_fake_binary_loss=binary_cross_entropy(fake_binary, real_vector)

            fake_binary_loss=binary_cross_entropy(fake_binary, fake_vector)
            real_binary_loss=binary_cross_entropy(real_binary,real_vector)
            if args.use_clip:

                fake_clip_style=clip_classifier(fake_images)
                #fake_clip_style=fake_clip_style.to(uniform.device)
                style_ambiguity_loss=cross_entropy(fake_clip_style, uniform)
                style_classification_loss=torch.tensor(0.)
            else:
                style_classification_loss=cross_entropy(real_style,real_labels)
                style_ambiguity_loss=cross_entropy(fake_style, uniform)

            style_classification_loss*=args.style_lambda
            style_ambiguity_loss*=args.style_lambda

            style_classification_loss_sum+=torch.sum(style_classification_loss)
            fake_binary_loss_sum+=torch.sum(fake_binary_loss)
            real_binary_loss_sum+=torch.sum(real_binary_loss)
            style_ambiguity_loss_sum+=torch.sum(style_ambiguity_loss)
            reverse_fake_binary_loss_sum+=torch.sum(reverse_fake_binary_loss)

            disc_loss=style_classification_loss+fake_binary_loss+real_binary_loss
            gen_loss=style_ambiguity_loss+reverse_fake_binary_loss



            accelerator.backward(gen_loss, retain_graph=True)
            gen_optimizer.step()

            

            accelerator.backward(disc_loss)
            disc_optimizer.step()
        accelerator.log({
            "style_classification":style_classification_loss_sum,
            "fake_binary": fake_binary_loss_sum,
            "real_binary":real_binary_loss_sum,
            "style_ambiguity":style_ambiguity_loss_sum,
            "reverse_fake_binary":reverse_fake_binary_loss_sum
        })

        end=time.time()
        print(f"epoch {e} elapsed {end-start} seconds")
        if e%10==0:
            checkpoint_dir=f"{args.output_dir}/checkpoint_{e}"
            os.makedirs(checkpoint_dir,exist_ok=True)
            torch.save(gen.state_dict(),f"{checkpoint_dir}/gen-weights.pickle")
            torch.save(disc.state_dict(),f"{checkpoint_dir}/disc-weights.pickle")
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message=f"epoch {e}",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()
    torch.save(gen.state_dict(),args.output_dir+"/gen-weights.pickle")
    torch.save(disc.state_dict(),args.output_dir+"/disc-weights.pickle")
    
    upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
    )
    model_card_content=f"""
    Creative Adversarial Network \n
    epochs: {args.epochs} 
    dataset {args.dataset_name}
    n classes {n_classes}
    batch_size {args.batch_size}
    images where resized to {args.resize_dim}
     and then center cropped to: {args.image_dim}
    used clip={args.use_clip}

    discriminator parameters:
    init_dim: {args.disc_init_dim}
    final_dim {args.disc_final_dim}

    generator parameters:
    input noise_dim: {args.gen_z_dim}
    """
    card=ModelCard(model_card_content)
    card.push_to_hub(repo_id)
    print("all done :)))")

if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    training_loop(args)