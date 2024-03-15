from dataclasses import dataclass
import torch
@dataclass
class TrainConfig:
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    dataset_dir = "./audios/"
    sample_size = 128
    train_batch_size = 1
    eval_batch_size = 2
    num_epochs = 60
    gradient_accumulation_steps = 1
    mixed_precision = "no" if device == "mps" else "fp16"
    learning_rate = 0.0001
    lr_warmup_steps = 500
    save_model_epochs = 20
    output_dir = "./output/"
    seed = 1234
