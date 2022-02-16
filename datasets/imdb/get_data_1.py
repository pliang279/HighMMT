import h5py
from typing import *
from torch.utils.data import Dataset, DataLoader

import json
from PIL import Image
from typing import *
import os


class IMDBDataset(Dataset):

    def __init__(self, file:h5py.File, start_ind:int, end_ind:int, vggfeature:bool=False) -> None:
        self.file = file
        self.start_ind = start_ind
        self.size = end_ind-start_ind
        self.vggfeature = vggfeature

    def __getitem__(self, ind):
        if not hasattr(self, 'dataset'):
            self.dataset = h5py.File(self.file, 'r')
        text = self.dataset["features"][ind+self.start_ind].reshape(-1,1)
        image = self.dataset["images"][ind+self.start_ind].transpose(1,2,0) if not self.vggfeature else \
            self.dataset["vgg_features"][ind+self.start_ind].reshape(-1,1)
        label = self.dataset["genres"][ind+self.start_ind]

        return text, image, label

    def __len__(self):
        return self.size


class IMDBDataset_robust(Dataset):

    def __init__(self, dataset, start_ind:int, end_ind:int) -> None:
        self.dataset = dataset
        self.start_ind = start_ind
        self.size = end_ind-start_ind

    def __getitem__(self, ind):
        text = self.dataset[ind+self.start_ind][0]
        image = self.dataset[ind+self.start_ind][1]
        label = self.dataset[ind+self.start_ind][2]

        return text, image, label

    def __len__(self):
        return self.size

def process_data(filename, path):
    data = {}
    filepath = os.path.join(path, filename)

    with Image.open(filepath+".jpeg") as f:
        image = np.array(f.convert("RGB"))
        data["image"] = image

    with open(filepath+".json", "r") as f:
        info = json.load(f)

        plot = info["plot"]
        data["plot"] = plot

    return data

def get_dataloader(path:str,test_path:str,num_workers:int=8, train_shuffle:bool=True, batch_size:int=40, vgg:bool=False, skip_process=False, no_robust=False)->Tuple[Dict]:
    train_dataloader = DataLoader(IMDBDataset(path, 0, 15552, vgg), \
        shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(IMDBDataset(path, 15552, 18160, vgg), \
        shuffle=False, num_workers=num_workers, batch_size=batch_size)
    if no_robust:
        test_dataloader = DataLoader(IMDBDataset(path, 18160, 25959, vgg), \
            shuffle=False, num_workers=num_workers, batch_size=batch_size)
        return train_dataloader,val_dataloader,test_dataloader


