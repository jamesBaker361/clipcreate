from better_pipeline import BetterDefaultDDPOStableDiffusionPipeline
from discriminator_src import Discriminator
from generator_src import Generator

def get_params(model)->int:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def print_children(model):
    for layer in model.children():
        if hasattr(layer, 'out_features'):
            print(layer.out_features)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

if __name__=='__main__':
    vanilla_pipeline=BetterDefaultDDPOStableDiffusionPipeline("stabilityai/stable-diffusion-2-base",use_lora=False)
    print("vae params ",print_trainable_parameters(vanilla_pipeline.sd_pipeline.vae))
    print("text encoder params ",print_trainable_parameters(vanilla_pipeline.sd_pipeline.text_encoder))
    print("unet params ", print_trainable_parameters(vanilla_pipeline.sd_pipeline.unet))

    vanilla_pipeline=BetterDefaultDDPOStableDiffusionPipeline("stabilityai/stable-diffusion-2-base",use_lora=True)
    print("unet lora params ", print_trainable_parameters(vanilla_pipeline.sd_pipeline.unet))
    #print("unet params trainable ", vanilla_pipeline.sd_pipeline.unet.print_trainable_parameters())

    disc=Discriminator(64, 32,512,["f{i}" for i in range(27)],False,True)
    print("disc params ", print_trainable_parameters(disc))
    print(disc.main)
    print_children(disc.main)
    print("binary")
    print(disc.binary_layers)
    print("style")
    print(disc.style_layers)



    gen=Generator(100,64,False)
    print("gen params ",print_trainable_parameters(gen))
    print(gen.main)
    print_children(gen.main)
