"""Ordered drug pairs with two molecular images and a paired motif-token sequence."""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DDIDataset(Dataset):
    """Returns (image_1, image_2, motif_sequence, attention_mask, label) for each ordered pair.

    The motif sequence of a pair is [drug 1 motifs (L)] + [SEP] + [drug 2 motifs (L)];
    each drug keeps its first L motif IDs and is right-padded with 0.
    """

    def __init__(self, image_folder, image_table, motif_vocab_file, split_file,
                 img_transformer=None, normalize=None, max_motif_len=8):
        print(f"[Dataset] Loading pairs from: {split_file}")
        self.image_folder = image_folder
        self.txt_file = pd.read_csv(image_table)
        self.normalize = normalize
        self._image_transformer = img_transformer
        self.ddi_data = pd.read_csv(split_file)
        self.max_motif_len = max_motif_len
        self.vocab = self.load_vocab(motif_vocab_file)
        self.vacb_size = len(self.vocab)
        substructure_file = os.path.join(os.path.dirname(motif_vocab_file), "drug_substructure_dict.npz")
        self.npz_data = np.load(substructure_file, allow_pickle=True)['drug_substructure_dict'].item()
        self.ddi_pairs = {}
        self.process_ddi_data()
        self.keys = list(self.ddi_pairs.keys())

    def load_vocab(self, motif_vocab_file):
        vocab = {}
        data = pd.read_csv(motif_vocab_file)
        for _, row in data.iterrows():
            vocab[row['motif']] = row['motif_id']
        return vocab

    def process_ddi_data(self):
        for _, row in self.ddi_data.iterrows():
            self.ddi_pairs[(row['drug_1'], row['drug_2'])] = row['label']

    def get_image(self, drug_id):
        try:
            if not (self.txt_file['drug_id'] == drug_id).any():
                print(f"Drug ID {drug_id} not found in the image table")
                return None
            image_name = self.txt_file[self.txt_file['drug_id'] == drug_id]['filename'].iloc[0]
            image_path = f"{self.image_folder}/{image_name}"
            img = Image.open(image_path).convert('RGB')
            if self._image_transformer:
                img = self._image_transformer(img)
            return img
        except Exception as e:
            print(f"Error in get_image: {e}")

    def __getitem__(self, index):
        drug_id1, drug_id2 = self.keys[index]
        label = self.ddi_pairs[(drug_id1, drug_id2)]

        img_1 = self.get_image(drug_id1)
        img_2 = self.get_image(drug_id2)
        if img_1 is None or img_2 is None:
            print(f"Image not found for drug_id1: {drug_id1} or drug_id2: {drug_id2}")
            return None
        if self.normalize:
            img_1 = self.normalize(img_1)
            img_2 = self.normalize(img_2)

        h_motifs = self.npz_data[drug_id1]
        t_motifs = self.npz_data[drug_id2]
        sep_token_id = self.vacb_size + 1
        pad_token_id = 0

        # Truncate or right-pad each drug to max_motif_len tokens.
        if len(h_motifs) > self.max_motif_len:
            h_motifs = h_motifs[:self.max_motif_len]
        else:
            h_motifs = list(h_motifs) + [pad_token_id] * (self.max_motif_len - len(h_motifs))
        if len(t_motifs) > self.max_motif_len:
            t_motifs = t_motifs[:self.max_motif_len]
        else:
            t_motifs = list(t_motifs) + [pad_token_id] * (self.max_motif_len - len(t_motifs))

        motif_seq = list(h_motifs) + [sep_token_id] + list(t_motifs)
        motif_seq = torch.tensor(motif_seq, dtype=torch.long)
        attention_mask = (motif_seq != pad_token_id).long()  # 1 for real tokens and SEP, 0 for padding
        return img_1, img_2, motif_seq, attention_mask, label

    def __len__(self):
        return len(self.ddi_pairs)


def get_datasets(dataset, dataroot):
    image_folder = os.path.join(dataroot, "drug_image", dataset)
    image_table = os.path.join(dataroot, "drug_image.csv")
    assert os.path.isdir(image_folder), "{} is not a directory.".format(image_folder)
    assert os.path.isfile(image_table), "{} is not a file.".format(image_table)
    return image_folder, image_table
