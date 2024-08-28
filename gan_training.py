import os
import torch
from generator_src import Generator
from discriminator_src import Discriminator,GANDataset,UtilDataset

from huggingface_hub import create_repo, upload_folder, ModelCard
from torchvision.transforms import ToPILImage
from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator
import wandb
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time
from static_globals import *
from scipy.special import softmax
import numpy as np
import datetime
import random
from count_params import print_trainable_parameters
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description="classifier training")
parser.add_argument("--epochs", type=int,default=50)
parser.add_argument("--dataset_name",type=str,default="jlbaker361/wikiart")
parser.add_argument("--batch_size",type=int, default=4)
parser.add_argument("--repo_id",type=str,default="jlbaker361/dcgan-wikiart")
parser.add_argument("--output_dir",type=str,default="/scratch/jlb638/dcgan-wikiart")
parser.add_argument("--use_clip",type=bool,default=False)
parser.add_argument("--use_kmeans",type=bool,default=False)
parser.add_argument("--center_list_path",type=str,default="test_centers.npy", help="path for np files that are centers of the clusters for k means")
parser.add_argument("--style_lambda",type=float,default=0.1, help="coefficient on style terms")
parser.add_argument("--conditional", type=bool,default=False,help="whether to use conditional GAN or not")

parser.add_argument("--gen_z_dim",type=int,default=100,help="dim latent noise for generator")
parser.add_argument("--image_dim", type=int,default=512)
parser.add_argument("--disc_init_dim",type=int,default=32)
parser.add_argument("--disc_final_dim",type=int,default=512)

parser.add_argument("--style_list",nargs="+",default=WIKIART_STYLES)
parser.add_argument("--resize_dim",type=int,default=768)

parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
parser.add_argument("--classifier_only",default=False,help="whether to only use style classifier")
parser.add_argument("--class_loss",type=str,default="cross_entropy")
parser.add_argument("--reverse_fake_binary_loss_weight",type=float,default=1.0)
parser.add_argument("--style_ambiguity_loss_weight",type=float,default=1.0)
parser.add_argument("--wasserstein",type=bool,default=False)
parser.add_argument("--n_disc_steps",type=int,default=1,help="how many extra times to train discriminatir")
parser.add_argument("--use_gp",action="store_true",help="whether to use gradient penalty for wasserstein")
parser.add_argument("--gp_weight",type=float,default=10)
parser.add_argument("--beta_0",type=float,default=0.5)
parser.add_argument("--project_name",type=str,default="creativity")
parser.add_argument("--n_test_images",type=int,default=3)
parser.add_argument("--load_from_output",action="store_true")

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def get_gradients(disc,fake_images,real_images,text_encoding):
    #https://necromuralist.github.io/Neurotic-Networking/posts/gans/wasserstein-gan-with-gradient-penalty/index.html
    epsilon= random.random()
    fake_images=epsilon*fake_images
    real_images=(1.0-epsilon) *real_images
    mixed_images=fake_images+real_images
    mixed_scores,_mixed_class = disc(mixed_images,text_encoding)
    gradients = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        inputs = mixed_images,
        outputs = mixed_scores,
        # These other parameters have to do with how the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradients

def get_gradient_penalty(gradients,gp_weight):

    gradients = gradients.view(len(gradients), -1)
    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()

def training_loop(args):
    softmax = torch.nn.Softmax(dim=1) # discriminator doesn't have softmax layer()
    loss_bce = torch.nn.BCELoss()
    loss_cls = torch.nn.CrossEntropyLoss()  # includes logSoftmax
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    device=accelerator.device
    print(f"acceleerate device = {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir,exist_ok=True)
    dataset=GANDataset(args.dataset_name,args.image_dim,args.resize_dim,args.batch_size,"train")
    style_list=list(dataset.style_set)
    print("style_list:")
    for s in style_list:
        print("\t",s)
    gen=Generator(args.gen_z_dim,args.image_dim,args.conditional)
    gen.apply(weights_init)
    print("gen main:")
    print(gen.main)
    print_trainable_parameters(gen.main)
    disc=Discriminator(args.image_dim, args.disc_init_dim,args.disc_final_dim,style_list,args.conditional,args.wasserstein)
    print("disc main")
    print(disc.main)
    print_trainable_parameters(disc.main)
    print("binary")
    print(disc.binary_layers)
    print_trainable_parameters(disc.binary_layers)
    print("style")
    print(disc.style_layers)
    print_trainable_parameters(disc.style_layers)

    start_epoch=0


    if os.path.exists(args.output_dir):
        checkpoints = list(
        filter(
            lambda x: "checkpoint_" in x,
            os.listdir(args.output_dir),
            )
        )
        if len(checkpoints) == 0:
            print(f"No checkpoints found in {args.output_dir}")
            disc.apply(weights_init)
            gen.apply(weights_init)
        elif args.load_from_output:
            checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
            print(f"loading from checkpoint_{checkpoint_numbers[-1]}")
            disc_weight_path = os.path.join(
                args.output_dir,
                f"checkpoint_{checkpoint_numbers[-1]}/disc-weights.pickle",
            )
            gen_weight_path = os.path.join(
                args.output_dir,
                f"checkpoint_{checkpoint_numbers[-1]}/gen-weights.pickle",
            )
            try:
                if torch.cuda.is_available():
                    disc_state_dict=torch.load(disc_weight_path)
                    gen_state_dict=torch.load(gen_weight_path)
                else:
                    disc_state_dict=torch.load(disc_weight_path,map_location=torch.device('cpu'))
                    gen_state_dict=torch.load(gen_weight_path,map_location=torch.device('cpu'))
                disc.load_state_dict(disc_state_dict)
                gen.load_state_dict(gen_state_dict)
                start_epoch=checkpoint_numbers[-1]+1
            except FileNotFoundError:
                print(f"couldn't find {disc_weight_path} or {gen_weight_path}")
                disc.apply(weights_init)
                gen.apply(weights_init)

            

    
    for x,y,z in dataset:
        break
    print(y.size())
    print(y.size()[-1])
    n_classes=y.size()[-1]
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",do_rescale=False)

    model,processor=accelerator.prepare(model,processor)
    freeze_model(model)
    if args.use_clip:
        print("using clip classifier")
        def clip_classifier(images):
            try:
                inputs = processor(images=images,text=style_list, return_tensors="pt", padding=True)
            except ValueError:
                images=images+1
                images=images/2
                inputs = processor(images=images,text=style_list, return_tensors="pt", padding=True)
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            inputs['pixel_values'] = inputs['pixel_values'].to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            return logits_per_image
        
    if args.use_kmeans:
        def cosine_similarity(vector1, vector2):
            """
            Calculate cosine similarity between two vectors.
            
            Args:
            vector1 (numpy array): First vector.
            vector2 (numpy array): Second vector.
            
            Returns:
            float: Cosine similarity between the two input vectors.
            """
            dot_product = np.dot(vector1, vector2)
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)
            
            if norm_vector1 == 0 or norm_vector2 == 0:
                return 0  # Return 0 if one of the vectors is a zero vector
            
            return dot_product / (norm_vector1 * norm_vector2)
        print("using kmeans classifier")
        center_list=np.load(args.center_list_path)
        n_classes=len(center_list)
        def kmeans_classifier(images):
            try:
                inputs = processor(images=images,text="text", return_tensors="pt", padding=True)
            except ValueError:
                images=images+1
                images=images/2
                inputs = processor(images=images,text="text", return_tensors="pt", padding=True)
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            inputs['pixel_values'] = inputs['pixel_values'].to(device)
            outputs = model(**inputs)
            image_embeds=outputs.image_embeds.cpu().numpy()
            y_pred_list=[]
            for x in image_embeds:
                y_pred=[]
                for center in center_list:
                    try:
                        dist=1.0/ np.linalg.norm(center - x)
                    except ZeroDivisionError:
                        dist=100000
                    y_pred.append(dist)
                #y_pred=softmax(y_pred)
                y_pred_list.append(y_pred)
            return torch.tensor(y_pred_list).to(device)
    util_dataset=UtilDataset(args.gen_z_dim, len(dataset),n_classes)
    try:
        repo_id=create_repo(repo_id=args.repo_id, exist_ok=True).repo_id
    except:
        print("retrying creating repo")
        repo_id=create_repo(repo_id=args.repo_id, exist_ok=True).repo_id

    gen_optimizer=optim.Adam(gen.parameters(),lr=0.0002,betas=(args.beta_0,0.999))
    disc_optimizer=optim.Adam(disc.parameters(),lr=0.0002,betas=(args.beta_0,0.999))
    print("len gen.parameters()",len([gp for gp in gen.parameters()]))
    print("len disc.parameters ", len([dp for dp in disc.parameters()]))
    #scheduler=optim.lr_scheduler.LinearLR(optimizer)
    training_dataloader=DataLoader(dataset, batch_size=args.batch_size,drop_last=True)
    util_dataloader=DataLoader(util_dataset,batch_size=args.batch_size,drop_last=True)

    #=dataset.sentence_trans
    #constant_noise=torch.randn(1,args.gen_z_dim, 1, 1)
    #constant_text_encoding=torch.tensor(sentence_trans.encode("painting"))
    gen, gen_optimizer, disc, disc_optimizer, training_dataloader, util_dataloader, = accelerator.prepare(gen, 
                                                                                                         gen_optimizer,
                                                                                                           disc, disc_optimizer, training_dataloader,
                                                                                                           util_dataloader)
    
    constant_noise_list=[]
    for x,(batch,util_vectors) in enumerate(zip(training_dataloader,util_dataloader)):
        if x>=args.n_test_images:
            break
        constant_noise,_real_vector,_fake_vector,_uniform = util_vectors
        _real_images, _real_labels,constant_text_encoding = batch
        constant_noise_list.append(constant_noise)


    classification_loss=torch.nn.CrossEntropyLoss()
    if args.class_loss=="mse":
        classification_loss=torch.nn.MSELoss()
    elif args.class_loss=="mae":
        classification_loss=torch.nn.L1Loss()
    elif args.class_loss=="softmax_ce":
        def _ce(x):
            x=torch.nn.Softmax(1)(x)
            return torch.nn.CrossEntropyLoss()(x)
        classification_loss=_ce
    binary_cross_entropy = torch.nn.BCELoss()
    def modified_generator_entropy(predictions,labels):
        #https://github.com/tensorflow/tensorflow/blob/2007e1ba474030fcce840b0b8a599558e7d5998f/tensorflow/contrib/gan/python/losses/python/losses_impl.py#L563
        #L = -log(sigmoid(D(G(z))))
        return
    real_label_int = 1.
    fake_label_int = 0.
    print(f"starting at epoch {start_epoch}")
    for e in range(start_epoch,args.epochs):
        D_x_binary_sum=0.0
        D_x_style_sum=0.0
        D_g_binary_sum=0.0
        G_g_binary_sum=0.0
        G_g_style_sum=0.0
        start=time.time()
        torch.cuda.empty_cache()
        accelerator.free_memory()
        for batch,util_vectors in zip(training_dataloader,util_dataloader):
            noise,real_vector,fake_vector,uniform = util_vectors
            noise=torch.rand(noise.size(),device=noise.device) #its possible these arent as random as they should be
            real_images, real_labels,text_encoding = batch

            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            real_binary,real_style=disc(real_images,text_encoding)
            #real_style=torch.nn.Softmax(1)(real_style)
            fake_images=gen(noise, text_encoding)
            fake_binary,fake_style=disc(fake_images.detach(),text_encoding)

            err_d_r=loss_bce(real_binary, real_vector)
            
            err_d_cls=loss_cls(real_style, real_labels)

            err = err_d_r + err_d_cls
            err.backward()

            D_x_binary = err_d_r.mean().item()
            D_g_binary_sum+=D_x_binary
            D_x_style=err_d_cls.mean().item()
            D_x_style_sum+=D_x_style

            err_d_f=loss_bce(fake_binary,fake_vector)
            err_d_f.backward()

            D_g_binary=fake_binary.mean().item()
            D_g_binary_sum+=D_g_binary

            disc_optimizer.step()

            #gen training?
            fake_binary,fake_style=disc(fake_images,text_encoding)
            err_g_r = loss_bce(fake_binary, real_vector)
            err_g_ambiguity = args.style_lambda * loss_bce( F.softmax(fake_style, dim=1), uniform)

            G_g_style=err_g_ambiguity.mean().item()
            G_g_style_sum+=G_g_style

            err_g = err_g_r + err_g_ambiguity
            err_g.backward()

            G_g_binary = err_g_r.mean().item()
            G_g_binary_sum+=G_g_binary
            gen_optimizer.step()


            accelerator.log({
                "Discriminator_real_binary":D_x_binary,
                "Discriminator_real_style":D_x_style,
                "Discriminator_fake_binary":D_g_binary,
                "Generator_fake_style":G_g_style,
                "Generator_fake_binary":G_g_binary
            })
            
            '''for _ in range(args.n_disc_steps):
                #real loss discirinator
                real_binary,real_style=disc(real_images,text_encoding)
                real_labels=real_labels.to(real_style.dtype)
                
                if args.use_clip or args.use_kmeans:
                    style_classification_loss=torch.tensor(0.)
                else:
                    style_classification_loss=classification_loss(real_style,real_labels)
                if args.classifier_only:
                    accelerator.backward(style_classification_loss)
                    disc_optimizer.step()
                    disc_optimizer.zero_grad()
                    style_classification_loss_sum+=torch.sum(style_classification_loss)
                    continue

                #fake image loss discrikinator
                fake_images=gen(noise, text_encoding)
                fake_binary,fake_style=disc(fake_images.detach(),text_encoding.detach())
                
                

                if args.wasserstein:
                    difference_loss=torch.mean(fake_binary-real_binary)
                    #https://github.com/tensorflow/gan/blob/656e4332d1e6d7f398f0968966c753e44397fc60/tensorflow_gan/python/losses/losses_impl.py#L111
                    disc_loss=style_classification_loss+difference_loss
                    if args.use_gp:
                        gradients=get_gradients(disc,fake_images,real_images,text_encoding)
                        gradient_penalty=get_gradient_penalty(gradients,args.gp_weight)
                        disc_loss+=gradient_penalty
                else:
                    fake_binary_loss=binary_cross_entropy(fake_binary, fake_vector)
                    real_binary_loss=binary_cross_entropy(real_binary,real_vector)
                    disc_loss=style_classification_loss+fake_binary_loss+real_binary_loss
                accelerator.backward(disc_loss)
                disc_optimizer.step()
                disc_optimizer.zero_grad()

            fake_images=gen(noise, text_encoding)
            fake_binary,fake_style=disc(fake_images,text_encoding)
            
            if args.wasserstein:
                reverse_fake_binary_loss=torch.mean(fake_binary)
            else:
                reverse_fake_binary_loss= binary_cross_entropy(fake_binary, real_vector)
            reverse_fake_binary_loss=args.reverse_fake_binary_loss_weight * reverse_fake_binary_loss
            if args.use_clip:
                fake_clip_style=clip_classifier(fake_images)
                style_ambiguity_loss=torch.tensor(classification_loss(fake_clip_style, uniform).cpu().numpy(),requires_grad=True)
            elif args.use_kmeans:
                fake_kmeans_style=kmeans_classifier(fake_images)
                style_ambiguity_loss=torch.tensor(classification_loss(fake_kmeans_style, uniform).cpu().numpy(),requires_grad=True)
            else:
                fake_binary,fake_style=disc(fake_images,text_encoding)
                style_ambiguity_loss=classification_loss(fake_style, uniform)

            style_ambiguity_loss=args.style_ambiguity_loss_weight * style_ambiguity_loss
            gen_loss=style_ambiguity_loss+reverse_fake_binary_loss
            accelerator.backward(gen_loss)
            gen_optimizer.step()
            gen_optimizer.zero_grad()

            
            style_classification_loss_sum+=torch.sum(style_classification_loss)
            if args.wasserstein:
                difference_loss_sum+=torch.sum(difference_loss)
            else:
                fake_binary_loss_sum+=torch.sum(fake_binary_loss)
                real_binary_loss_sum+=torch.sum(real_binary_loss)
            style_ambiguity_loss_sum+=torch.sum(style_ambiguity_loss)
            reverse_fake_binary_loss_sum+=torch.sum(reverse_fake_binary_loss)'''
        for i in range(args.n_test_images):
            test_image=gen(constant_noise_list[i], constant_text_encoding)
            pil_test_image=ToPILImage()( test_image[0])
            path=f"tmp_{i}.png"
            pil_test_image.save(path)
            try:
                accelerator.log({
                    f"test_image_{i}":wandb.Image(path)
                })
            except:
                accelerator.log({
                    f"test_image_{i}":wandb.Image(pil_test_image)
                })

        accelerator.log({
                "Discriminator_real_binary_sum":D_x_binary_sum,
                "Discriminator_real_style_sum":D_x_style_sum,
                "Discriminator_fake_binary_sum":D_g_binary_sum,
                "Generator_fake_style_sum":G_g_style_sum,
                "Generator_fake_binary_sum":G_g_binary_sum,
            })

        end=time.time()
        print(f"epoch {e} elapsed {end-start} seconds")
        if e%5==0:
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
        accelerator.free_memory()
        torch.cuda.empty_cache()
    tracker_url=accelerator.get_tracker("wandb").run.get_url()
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
    conditional ={args.conditional}

    discriminator parameters:
    init_dim: {args.disc_init_dim}
    final_dim {args.disc_final_dim}

    generator parameters:
    input noise_dim: {args.gen_z_dim}
    wandb project: {tracker_url}
    """
    card=ModelCard(model_card_content)
    card.push_to_hub(repo_id)
    print("all done :)))")

if __name__=='__main__':
    args=parser.parse_args()
    for slurm_var in ["SLURMD_NODENAME","SBATCH_CLUSTERS", 
                      "SBATCH_PARTITION","SLURM_JOB_PARTITION",
                      "SLURM_NODEID","SLURM_MEM_PER_GPU",
                      "SLURM_MEM_PER_CPU","SLURM_MEM_PER_NODE","SLURM_JOB_ID"]:
        try:
            print(slurm_var, os.environ[slurm_var])
        except:
            print(slurm_var, "doesnt exist")
    try:
        print('torch.cuda.get_device_name()',torch.cuda.get_device_name())
    except Exception as e:
        print("couldnt print torch.cuda.get_device_name()")
        print(e)
    try:
        print('torch.cuda.get_device_capability()',torch.cuda.get_device_capability())
    except Exception as e:
        print("couldnt print torch.cuda.get_device_capability()")
        print(e)
    try:
        current_device = torch.cuda.current_device()
    except Exception as e:
        print("couldnt get torch.cuda.current_device()")
        print(e)
    try:
        gpu = torch.cuda.get_device_properties(current_device)
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Memory Total: {gpu.total_memory / 1024**2} MB")
        print(f"GPU Memory Free: {torch.cuda.memory_allocated(current_device) / 1024**2} MB")
        print(f"GPU Memory Used: {torch.cuda.memory_reserved(current_device) / 1024**2} MB")
    except Exception as e:
        print("couldnt get gpu properties")
        print(e)
    current_date_time = datetime.datetime.now()
    formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Formatted Date and Time:", formatted_date_time)
    print(args)
    training_loop(args)