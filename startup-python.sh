yes | conda create -n creativity python=3.11
conda init
conda activate creativity
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install wandb huggingface_hub peft diffusers bitsandbytes scipy matplotlib trl
#python3 -m pip install git+https://github.com/jamesBaker361/trl-fixed.git
