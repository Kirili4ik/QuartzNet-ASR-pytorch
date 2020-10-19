import torch
import torchaudio
import string
import pandas as pd
import math

from torch import distributions
from torch.nn.utils.rnn import pad_sequence


from utils import TextTransform

class TrainDataset(torch.utils.data.Dataset):
    """Custom competition dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.answers = pd.read_csv(csv_file, '\t')
        self.transform = transform


    def __len__(self):
        return len(self.answers)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        utt_name = 'cv-corpus-5.1-2020-06-22/en/clips/' + self.answers.loc[idx, 'path']
        utt = torchaudio.load(utt_name)[0].squeeze()
        if len(utt.shape) != 1:
            utt = utt[1]

        answer = self.answers.loc[idx, 'sentence']

        if self.transform:
            utt = self.transform(utt)

        sample = {'utt': utt, 'answer': answer}
        return sample


class TestDataset(torch.utils.data.Dataset):
    """Custom test dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.names = pd.read_csv(csv_file, '\t')
        self.transform = transform


    def __len__(self):
        return len(self.names)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        utt_name = 'cv-corpus-5.1-2020-06-22/en/clips/' + self.names.loc[idx, 'path']
        utt = torchaudio.load(utt_name)[0].squeeze()

        if self.transform:
            utt = self.transform(utt)

        sample = {'utt': utt}
        return sample


#win_len=1024, hop_len=256
# counting len of MelSpec before doing it (cause of padding)
def mel_len(x):
    return int(x // 256) + 1


def transform_tr(wav):
    aug_num = torch.randint(low=0, high=3, size=(1,)).item()
    augs = [
        lambda x: x,
        lambda x: (x + distributions.Normal(0, 0.01).sample(x.size())).clamp_(-1, 1),
        lambda x: torchaudio.transforms.Vol(.1)(x)
    ]
    return augs[aug_num](wav)


# collate_fn
def preprocess_data(data):
    text_transform = TextTransform()
    wavs = []
    input_lens = []
    labels = []
    label_lens = []

    for el in data:
        wavs.append(el['utt'])
        input_lens.append(math.ceil(mel_len(el['utt'].shape[0]) / 2))    # cause of stride 2
        label = torch.Tensor(text_transform.text_to_int(el['answer']))
        labels.append(label)
        label_lens.append(len(label))

    wavs = pad_sequence(wavs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    return wavs, input_lens, labels, label_lens
