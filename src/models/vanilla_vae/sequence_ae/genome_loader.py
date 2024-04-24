import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from genome_extractor import GenomeExtractor
from onehot import onehot_encoder

class LoaderConfig:
    batch_size: int = 50
    shuffle: bool = True
    num_workers: int = 1
    
class DataSet(Dataset):
    def __init__(self):
        pass

        
    def __getitem__(self, index):
        # label is the same as the original input
        return self.labels[index]
    
    def __len__(self):
        return len(self.labels)
    

config = LoaderConfig()
PATH = os.path.dirname(os.path.abspath("__file__"))
print(PATH)

train_dataset = DataSet(data_path = os.path.join(PATH, 'vanilla-vae', 'sequence_data/ohe_sequence_train.npy'))
test_dataset = DataSet(data_path = os.path.join(PATH, 'vanilla-vae', 'sequence_data/ohe_sequence_test.npy'))
val_dataset = DataSet(data_path = os.path.join(PATH, 'vanilla-vae', 'sequence_data/ohe_sequence_val.npy'))

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=config.shuffle,
    num_workers=config.num_workers
)

val_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=config.shuffle,
    num_workers=config.num_workers
)

test_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=config.shuffle,
    num_workers=config.num_workers
)