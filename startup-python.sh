# before this you shoudl run:
# yes | conda create -n creativity python=3.11
# conda init
# conda activate creativity

export BNB_CUDA_VERSION=118
yes |  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
yes |  conda install -c conda-forge  scipy matplotlib datasets
yes |  conda install -c conda-forge transformers
yes |  conda install -c conda-forge huggingface_hub
yes |  conda install -c conda-forge diffusers
yes |  conda install -c conda-forge wandb
#yes |  conda install -c conda-forge xformers
yes |  conda install -c conda-forge bitsandbytes
yes |  conda install -c conda-forge peft
yes |  conda install -c conda-forge sentence-transformers
#pip install trl["diffusers"]
pip install git+https://github.com/jamesBaker361/trl-fixed.git
# huggingface-cli login
# wandb login