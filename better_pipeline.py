from trl import DefaultDDPOStableDiffusionPipeline
from peft import LoraConfig
import torch

class BetterDefaultDDPOStableDiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        if self.use_lora:
            self.sd_pipeline.unet.requires_grad_(False)
            lora_config = LoraConfig(
                r=4,
                lora_alpha=4,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.sd_pipeline.unet.add_adapter(lora_config)

            # To avoid accelerate unscaling problems in FP16.
            for param in self.sd_pipeline.unet.parameters():
                # only upcast trainable parameters (LoRA) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)

    def get_trainable_layers(self):
        return self.sd_pipeline.unet