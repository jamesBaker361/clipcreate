# before this you shoudl run:
# yes | conda create -n creativity python=3.11
# conda init
# conda activate creativity

yes |  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
yes |  conda install -c conda-forge  scipy matplotlib datasets
yes |  conda install -c conda-forge transformers
yes |  conda install -c conda-forge huggingface_hub
yes |  conda install -c conda-forge diffusers
yes |  conda install -c conda-forge wandb
yes |  conda install -c conda-forge bitsandbytes
yes |  conda install -c conda-forge peft
pip install trl["diffusers"]
