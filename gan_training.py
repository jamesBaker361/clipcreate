import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from generator_src import Generator
from discriminator_src import Discriminator,GANDataset

from huggingface_hub import create_repo, upload_folder, ModelCard
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

parser.add_argument("--gen_z_dim",type=int,default=100,help="dim latent noise for generator")
parser.add_argument("--image_dim", type=int,default=512)
parser.add_argument("--disc_init_dim",type=int,default=32)
parser.add_argument("--disc_final_dim",type=int,default=512)

parser.add_argument("--style_list",nargs="+",default=WIKIART_STYLES)
parser.add_argument("--resize_dim",type=int,default=512)

def training_loop(args):

    os.makedirs(args.output_dir,exist_ok=True)
    gen=Generator(args.gen_z_dim,args.image_dim)
    disc=Discriminator(args.image_dim, args.disc_init_dim,args.disc_final_dim,args.style_list)
    dataset=GANDataset(args.dataset_name,args.image_dim,args.resize_dim,args.batch_size,"train")
    for x,y in dataset:
        break
    print(y.size())
    print(y.size()[-1])
    n_classes=y.size()[-1]

    gen_optimizer=optim.Adam(gen.parameters())
    disc_optimizer=optim.Adam(disc.parameters())
    #scheduler=optim.lr_scheduler.LinearLR(optimizer)
    training_dataloader=DataLoader(dataset, batch_size=args.batch_size,drop_last=True)

    accelerator = Accelerator()
    gen, gen_optimizer, disc, disc_optimizer, training_dataloader = accelerator.prepare(gen, gen_optimizer, disc, disc_optimizer, training_dataloader)
    cross_entropy=torch.nn.CrossEntropyLoss()
    binary_cross_entropy = torch.nn.BCELoss()
    uniform=torch.full((args.batch_size, n_classes), fill_value=1.0/n_classes)
    real_label_int = 1.
    fake_label_int = 0.
    for e in range(args.epochs):
        total_loss=0.0
        start=time.time()
        for batch in training_dataloader:
            real_images, real_labels = batch
            real_labels=real_labels.to(torch.float64)
            noise= torch.randn(args.batch_size, 100, 1, 1)
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            fake_images=gen(noise)
            real_binary,real_style=disc(real_images)
            fake_binary,fake_style=disc(fake_images)

            real_vector=torch.full((args.batch_size,1),fill_value=real_label_int)
            fake_vector=torch.full((args.batch_size,1),fill_value=fake_label_int)

            fake_binary_loss=binary_cross_entropy(fake_binary, fake_vector)
            reverse_fake_binary_loss=binary_cross_entropy(fake_binary, real_vector)
            real_binary_loss=binary_cross_entropy(real_binary,real_vector)
            style_classification_loss=cross_entropy(real_style,real_labels)
            style_ambiguity_loss=cross_entropy(fake_style, uniform)

            gen_loss=None
            disc_loss=style_classification_loss+fake_binary_loss+real_binary_loss
            gen_loss=style_ambiguity_loss+reverse_fake_binary_loss
            accelerator.backward(gen_loss, retain_graph=True)
            accelerator.backward(disc_loss)
            gen_optimizer.step()
            disc_optimizer.step()
            break

        end=time.time()
        print(f"epoch {e} elapsed {end-start} seconds")
    
    torch.save(gen.state_dict(),args.output_dir+"/gen-weights.pickle")
    torch.save(disc.state_dict(),args.output_dir+"/disc-weights.pickle")
    repo_id=create_repo(repo_id=args.repo_id, exist_ok=True).repo_id
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