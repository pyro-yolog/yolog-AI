from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import ClapModel
import torch.nn as nn
from diffusers import MusicLDMPipeline
import torch

def VAEModel(config):
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        sample_size=config.sample_size,
        block_out_channels=(128, 256, 384, 640),
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=(
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        latent_channels=16,
    )
    return model


def UNet2DCondition(config):
    model = UNet2DConditionModel(
        sample_size=config.sample_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=4,
        block_out_channels=(128, 256, 384, 640),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type=(
            "MidBlock2D",
        ),
        up_block_type=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


def CLAPModel():
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    return model

def HifiGAN(spectrogram):

    return model