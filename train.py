from DataLoader import DataLoad
from Models import *
from accelerate import Accelerator

from tqdm.auto import tqdm
import torch
import os
import torch.nn as nn

def VAEtrain(config):
    train_dataloader = DataLoad(config)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    accelerator.init_trackers("VAETrain", config=config, init_kwargs={"wandb": {"entity": "rlarmsgk2"}})
    model = VAEModel(config).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=4.5e-8)
    train_loss = 0
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for batch_idx, (data, _) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                data = data.to(config.device)
                optimizer.zero_grad()
                # total_params = sum(p.numel() for p in model.parameters())
                # print(total_params)
                recon_batch = model(data).sample
                criterion = nn.MSELoss()
                loss = criterion(recon_batch, data)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                train_loss += loss.item()
                optimizer.step()

            progress_bar.update(1)
            log = {'log': loss.item()}
            progress_bar.set_postfix(**log)
            accelerator.log(log, step=global_step)
            global_step += 1
    torch.save(model.state_dict(), './VAEModel.pt')
    accelerator.end_training()


# def display_mel():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import librosa
#     plt.figure(figsize=(7, 7))
#     ax = plt.subplot(2, 1, 1)
#     mel = np.load('./audios/Tailand shopping_mel.npy')
#     librosa.display.specshow(librosa.amplitude_to_db(mel, ref=0.00002), sr=44100, x_axis='time')
#     plt.colorbar(format='%2.0f dB')
#     plt.colorbar()
#     plt.show()