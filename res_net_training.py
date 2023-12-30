import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from res_net_src import ResNet, HFDataset
from huggingface_hub import create_repo, upload_folder, ModelCard
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time

parser = argparse.ArgumentParser(description="ddpo training")
parser.add_argument("--epochs", type=int,default=100)
parser.add_argument("--dataset_name",type=str,default="jlbaker361/wikiart")
parser.add_argument("--pretrained_version",type=str, default="resnet18")
parser.add_argument("--batch_size",type=int, default=4)
parser.add_argument("--repo_id",type=str,default="jlbaker361/resnet-wikiart")
parser.add_argument("--output_dir",type=str,default="/scratch/jlb638/resnet-wikiart")
def training_loop(epochs:int, dataset_name:str, pretrained_version:str,batch_size:int):
    hf_dataset=HFDataset(dataset_name,"train")
    for x,y in hf_dataset:
        break
    print(y.size())
    print(y.size()[-1])
    n_classes=y.size()[-1]
    model=ResNet(pretrained_version, n_classes)
    optimizer=optim.Adam(model.parameters())
    scheduler=optim.lr_scheduler.LinearLR(optimizer)
    training_dataloader=DataLoader(hf_dataset, batch_size=batch_size,drop_last=True)

    accelerator = Accelerator()
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, scheduler
    )
    criterion=torch.nn.CrossEntropyLoss()
    for e in range(epochs):
        total_loss=0.0
        start=time.time()
        for batch in training_dataloader:
            inputs, labels = batch
            labels=labels.to(torch.float64)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            #print('outputs',outputs.dtype, outputs.size())
            #print('labels',labels.dtype,labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            flat_loss=torch.flatten(loss)
            for fl in flat_loss:
                total_loss+=fl


        end=time.time()
        print(f"epoch {e} elapsed {end-start} seconds with total loss {total_loss}")
    torch.save(model.state_dict(),args.output_dir+"/resnet-weights.pickle")
    repo_id=create_repo(repo_id=args.repo_id).repo_id
    upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
    )
    model_card_content=f"""
    trained to classify
    epochs: {args.epochs} \n
    dataset {args.dataset_name} \n
    n classes {n_classes} \n
    pretrained version {args.pretrained_version} \n
    batch_size {args.batch_size}
    """
    card=ModelCard(model_card_content)
    card.push_to_hub(repo_id)


if __name__=="__main__":
    args=parser.parse_args()
    training_loop(args.epochs, 
                  args.dataset_name,
                  args.pretrained_version,
                  args.batch_size)