import torch
import torch.nn.functional as F
import wandb
import numpy as np

from tqdm import tqdm

from utils import cer, wer, decoder_func, beam_search_decoding


def train_epoch(model, optimizer, dataloader, CTCLoss, device, melspec_transforms):
    model.train()
    losses = []

    for i, (wavs, wavs_len, answ, answ_len) in tqdm(enumerate(dataloader)):
        wavs, answ = wavs.to(device), answ.to(device)

        trans_wavs = torch.log(melspec_transforms(wavs) + 1e-9)

        optimizer.zero_grad()

        output = model(trans_wavs)
        output = F.log_softmax(output, dim=1)
        output = output.transpose(0, 1).transpose(0, 2)

        loss = CTCLoss(output, answ, wavs_len, answ_len)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
        optimizer.step()
        losses.append(loss.item())
        if i % 100 == 0:
            wandb.log({'mean_train_loss':loss})
            preds, targets = decoder_func(output, answ, answ_len, del_repeated=False)
            wandb.log({"CER_train": cer(targets[0], preds[0])})
            wandb.log({"WER_train": wer(targets[0], preds[0])})

    return np.mean(losses)


def train(model, opt, train_dl, scheduler, CTCLoss, device, n_epochs, val_dl=None,
                                                    melspec=None, melspec_transforms=None):
    for epoch in range(n_epochs):
        print("Epoch {} of {}".format(epoch, n_epochs), 'LR', scheduler.get_last_lr())

        mean_loss = train_epoch(model, opt, train_dl, CTCLoss, device, melspec_transforms)
        print('MEAN EPOCH LOSS IS', mean_loss)

        scheduler.step()

        if (val_dl != None):
            test(model, opt, val_dl, CTCLoss, device, melspec=melspec)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, 'epoch_0_and_'+str(epoch))


def test(model, optimizer, dataloader, CTCLoss, device, melspec, bs_width=None):
    model.eval()

    cers, wers, cers_bs, wers_bs = [], [], [], []
    losses = []

    with torch.no_grad():
        for i, (wavs, wavs_len, answ, answ_len) in enumerate(dataloader):
            wavs, answ = wavs.to(device), answ.to(device)

            trans_wavs = torch.log(melspec(wavs) + 1e-9)

            output = model(trans_wavs)
            if bs_width != None:
                output_bs = F.softmax(output, dim=1).transpose(0, 1).transpose(0, 2)
                preds_bs, targets_bs = beam_search_decoding(output_bs, answ, answ_len, width=bs_width)

            output = F.log_softmax(output, dim=1)
            output = output.transpose(0, 1).transpose(0, 2)
            loss = CTCLoss(output, answ, wavs_len, answ_len)
            losses.append(loss.item())

            # argmax
            preds, targets = decoder_func(output, answ, answ_len, del_repeated=True)

            for i in range(len(preds)):
                if i == 0:
                    print('target: ', ''.join(targets[i]))
                    print('prediction: ', ''.join(preds[i]))

                cers.append(cer(targets[i], preds[i]))
                wers.append(wer(targets[i], preds[i]))
                if bs_width != None and i == 0:
                    print('beamS pred:', ''.join(preds_bs[i]))
                    cers_bs.append(cer(targets_bs[i], preds_bs[i]))
                    wers_bs.append(wer(targets_bs[i], preds_bs[i]))

        avg_cer = np.mean(cers)
        avg_wer = np.mean(wers)
        if bs_width != None:
            avg_cer_bs = np.mean(cers_bs)
            avg_wer_bs = np.mean(wers_bs)

        wandb.log({"CER_val": avg_cer})
        wandb.log({"WER_val": avg_wer})
        avg_loss= np.mean(losses)
        print('average test loss is', avg_loss)
        wandb.log({'mean_VAL_loss':avg_loss})
