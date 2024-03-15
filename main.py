from transformers import AutoProcessor
from config import TrainConfig
from train import VAEtrain
from Models import *
import torch
config = TrainConfig()
VAEtrain(config)

# repo_id = "ucsd-reach/musicldm"
# pipe = MusicLDMPipeline(vae=VAEModel(config), text_encoder=CLAPModel(), ).from_pretrained(repo_id, torch_dtype=torch.float16)
# pipe = pipe.to(config.device)

