import os
import sys
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset,load_dataset

class ResNet(torch.nn.Module):
    def __init__(self,pretrained_version,n_classes):
        '''The `__init__` function initializes a ResNet model with a specified pretrained version and number of
        classes, and sets up the model's layers and softmax activation.
        
        Parameters
        ----------
        pretrained_version
            The `pretrained_version` parameter is the version of the ResNet model that you want to load. It
        should be a string that specifies the ResNet architecture, such as "resnet18", "resnet[34,50,101,152]",
        "resnet50", etc. These architectures are available in the torchvision library
        n_classes
            The parameter `n_classes` represents the number of classes in the classification task. It is used
        to determine the number of output units in the final fully connected layer (`self.head`) of the
        ResNet model.
        
        '''
        super(ResNet, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', pretrained_version, pretrained=True)
        pretrained_layer_list=[m for m in model.children()][:-1]
        self.pretrained_sequential=torch.nn.Sequential(*pretrained_layer_list)
        self.n_classes=n_classes
        self.head=torch.nn.Linear(512, n_classes)
        self.softmax=torch.nn.Softmax(dim=-1)


    def forward(self,x):
        x=self.pretrained_sequential(x)
        x=x.view(x.size(0), -1)
        x=self.head(x)
        x=self.softmax(x)

        return x

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset_src="jlbaker361/wikiart",split="train",center_crop_dim=224,batch_size=4):

        hf_dataset=load_dataset(hf_dataset_src,split=split)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(center_crop_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data = [img for img in hf_dataset["image"]]
        self.data=[self.transform(img) for img in self.data]
        self.batch_size=batch_size
        limit=(len(self.data) // batch_size) *batch_size
        self.data=self.data[:limit]
        style_set=set([style for style in hf_dataset["style"]])
        encoding_dict={
            style:[0.0 for _ in range(len(style_set))] for style in style_set
        }
        for i,style in enumerate(style_set):
            encoding_dict[style][i]=1.0
        targets=[encoding_dict[style] for style in hf_dataset["style"]][:limit]
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index):

        x = self.data[index]
        try:
            x = self.transform(x)
        except RuntimeError:
            print(f"runtime error for index {index}  {len(self.data)}")
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)