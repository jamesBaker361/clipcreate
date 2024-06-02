import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torchvision import transforms
from sentence_transformers import SentenceTransformer
from datasets import Dataset,load_dataset
import torchvision.transforms.functional as functional

class Discriminator(nn.Module):
    def __init__(self, image_dim,init_dim,final_dim,style_list,conditional,wasserstein):
        super(Discriminator, self).__init__()
        self.image_dim=image_dim
        self.init_dim=init_dim
        self.final_dim=final_dim
        self.conditional=conditional
        layers=[
            nn.Conv2d(3, init_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        n_head_dim=init_dim*image_dim*image_dim//4
        '''while init_dim<final_dim and image_dim>4:
            interm_dim=int(3*init_dim/2)
            later_dim=init_dim*2
            layers.append(nn.Conv2d(init_dim, interm_dim, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(interm_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(interm_dim, later_dim, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(later_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            init_dim=later_dim
            image_dim//=4'''
        intermediate={64:2,128:3,256:4,512:5}
        for _ in range(intermediate[image_dim]):
            print(init_dim,image_dim)
            interm_dim=int(3*init_dim/2)
            later_dim=min(init_dim*2,512)
            layers.append(nn.Conv2d(init_dim, later_dim, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(later_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            init_dim=later_dim
            image_dim//=2
            n_head_dim//=2
        print(init_dim,image_dim)
        for _ in range(2):
            layers.append(nn.Conv2d(later_dim, later_dim, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(later_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Flatten())
        self.main = nn.Sequential(
            * layers
        )
        #n_head_dim=final_dim*image_dim*image_dim
        if self.conditional:
            self.conditional_linear=nn.Linear(768,n_head_dim)
            self.conditional_mha=nn.MultiheadAttention(n_head_dim, 4) #query = image kv=text
        if wasserstein:
            self.binary_layers=nn.Sequential(
                nn.Linear(n_head_dim,1),
                #nn.Sigmoid()
            )
        else:
            self.binary_layers=nn.Sequential(
                nn.Linear(n_head_dim,1),
                nn.Sigmoid()
            )
        self.style_layers=nn.Sequential(
            nn.Linear(n_head_dim,1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024,512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, len(style_list)),
            #nn.Softmax(dim=-1)
        )

    def forward(self, input,text_encoding):
        if self.conditional:
            main_output = self.main(input)
            text_output=self.conditional_linear(text_encoding)
            main_output=self.conditional_mha(main_output, text_output, text_output)
            (main_output, attn_output_weights)=main_output
        else:
            main_output = self.main(input)
        return self.binary_layers(main_output), self.style_layers(main_output)

class SquarePad:
    def __call__(self, image):
        w, h = image.size()[-2:]
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return functional.pad(image, padding, 0, 'constant')

class MakeRGB:
    def __call__(self,image):
        return image.convert("RGB")
    
class Rescale:
    def __call__(self,image):
        return (image-0.5)*2
    
class ImageCenterCrop:
    def __call__(self,image):
        width, height = image.size
        min_dimension = min(width, height)
        return transforms.CenterCrop(min_dimension)(image)

class GANDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset_src,image_dim,resize_dim,batch_size,split):

        hf_dataset=load_dataset(hf_dataset_src,split=split)

        self.transform = transforms.Compose([
            MakeRGB(),
            transforms.ToTensor(),
            SquarePad(),
            transforms.Resize(resize_dim),
            transforms.CenterCrop(image_dim),
            Rescale()
        ])
        self.sentence_trans=SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
        self.data = []
        for i,(img, text) in enumerate(zip(hf_dataset["image"],hf_dataset["text"])):
            try:
                trans_img=self.transform(img)
                del trans_img
                self.data.append(img)
            except RuntimeError:
                print(f"runtime error for image {i} / {len(self.data)}")
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
        self.text_encoding_data=[torch.Tensor(self.sentence_trans.encode(t)) for t in hf_dataset["text"]][:limit]
        
    def __getitem__(self, index):

        x = self.data[index]
        x=self.transform(x)
        y = self.targets[index]
        z = self.text_encoding_data[index]
        
        return x, y, z
    
    def __len__(self):
        return len(self.data)
    
class UtilDataset(torch.utils.data.Dataset):
    def __init__(self,noise_dim, len_data,n_classes):
        self.noise_dim=noise_dim
        self.len_data=len_data
        self.n_classes=n_classes

    def __getitem__(self, index):
        #noise,real_vector,fake_vector,uniform
        return torch.randn(self.noise_dim, 1, 1),torch.tensor([1.]),torch.tensor([0.]),torch.tensor([1.0/self.n_classes for _ in range(self.n_classes)])
    
    def __len__(self):
        return self.len_data