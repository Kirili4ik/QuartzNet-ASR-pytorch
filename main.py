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
from train_test import train


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
    BATCH_SIZE = 80
    NUM_EPOCHS = 5
    N_MELS     = 64

    set_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Loading data and loaders
    my_dataset = TrainDataset(csv_file='train_preprocessed.tsv', transform=transform_tr)
    test_dataset = TestDataset(csv_file='cv-corpus-5.1-2020-06-22/en/test.tsv', transform=None)
    # sorted indexes
    with open('sorted.npy', 'rb') as f:
        s = np.load(f)
    to_save = s[:100000][:, 0]
    my_dataset = torch.utils.data.Subset(my_dataset, to_save)
    train_set, val_set = torch.utils.data.random_split(my_dataset, [85000, 15000])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=preprocess_data, drop_last=True,
                          num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=preprocess_data, drop_last=True,
                          num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

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
    # with augmentations
    melspec_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256,  n_mels=N_MELS),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35),
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
    CTCLoss = nn.CTCLoss(blank=0).to(device)
    train(model, opt, train_loader, scheduler, CTCLoss, device,
                      n_epochs=NUM_EPOCHS, val_dl=val_loader,
                      melspec=melspec, melspec_transforms=melspec_transforms)

