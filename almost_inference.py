import torch
import torch.nn as nn
import torchaudio
import random
import numpy as np
import pandas as pd
import wandb
import torch_optimizer

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# project imports
from my_datasets import TrainDataset, TestDataset, mel_len, preprocess_data, transform_tr
from model import QuartzNet
from train_test import test


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


if __name__ == '__main__':
    BATCH_SIZE = 10
    N_MELS     = 64

    set_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Loading data and loaders
    my_dataset = TrainDataset(csv_file='train_preprocessed.tsv', transform=transform_tr)
    # sorted indexes
    with open('sorted.npy', 'rb') as f:
        s = np.load(f)
    to_save = s[200:300][:, 0]
    val_set = torch.utils.data.Subset(my_dataset, to_save)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=preprocess_data, drop_last=True,
                          num_workers=0, pin_memory=True)

    ### wandb logins
    wandb.login()
    wandb.init()
    train_table = wandb.Table(columns=["Predicted Text", "True Text"])

    ### Creating melspecs on GPU
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,           ### 22050, 48000
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS                ### 64,    80
    ).to(device)

    ### Creating model from scratch
    model = QuartzNet(n_mels=64, num_classes=28)
    print('num of params', count_parameters(model))
    model.to(device)
    wandb.watch(model)
    opt = torch_optimizer.NovoGrad(
                        model.parameters(),
                        lr=0.01,
                        betas=(0.8, 0.5),
                        weight_decay=0.001,
    )
    scheduler  = CosineAnnealingLR(opt, T_max=50, eta_min=0, last_epoch=-1)

    # loading checkpoint
    checkpoint = torch.load('epoch_5', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    CTCLoss = nn.CTCLoss(blank=0).to(device)
    test(model, opt, val_loader, CTCLoss, device, bs_width=8, melspec=melspec)

