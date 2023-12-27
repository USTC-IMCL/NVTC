import json
import glob
import io
import lmdb
from pathlib import Path
import imageio.v2 as iio
from PIL import Image
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning.pytorch as pl


class ImageLMDBDataset(Dataset):
    def __init__(self, root_dir, crop_size, name=''):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.name = name
        # self.env = lmdb.open(root_dir, readonly=True, lock=False,
        #                      readahead=False, meminit=False)
        # self.txn = self.env.begin(write=False)
        # self.n_samples = int(self.txn.get('num-_samples'.encode()))

        # Delay loading LMDB data until after initialization to avoid "can't
        # pickle Environment Object error"
        self.env, self.txn = None, None
        with lmdb.open(root_dir, readonly=True, lock=False, readahead=False, meminit=False) as env:
            with env.begin(write=False) as txn:
                self.n_samples = int(txn.get('num-samples'.encode()))

        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])

    def _init_db(self):
        self.env = lmdb.open(self.root_dir, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization:
        # https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        img_key = 'img-{:0>9}'.format(index + 1)
        img_bin = self.txn.get(img_key.encode())
        img = Image.open(io.BytesIO(img_bin))
        img = img.convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return self.n_samples


class ImageGlobDataset(Dataset):
    def __init__(self, glob_p, crop_size=None, with_name=False):
        self.img_ps = sorted(glob.glob(glob_p))
        self.with_name = with_name
        if crop_size is not None:
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.img_ps)

    def __getitem__(self, index):
        img_p = self.img_ps[index]
        img = Image.open(img_p).convert("RGB")
        img = self.transform(img)
        if self.with_name:
            img_name = Path(img_p).stem
            return img, img_name
        return img


class ImageCodingDataModule(pl.LightningDataModule):
    def __init__(self, train_dir: str, test_glob: str, crop_size: int, batch_size: int,
                 num_workers: int):
        super().__init__()
        self.train_dir = train_dir
        self.test_glob = test_glob
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        pass

    def setup(self, stage):
        if stage == "fit":
            self.train_set = ImageLMDBDataset(self.train_dir, self.crop_size)
            self.eval_set = ImageGlobDataset(self.test_glob)
        elif stage == "validate":
            self.eval_set = ImageGlobDataset(self.test_glob)
        elif stage == "test":
            self.test_set = ImageGlobDataset(self.test_glob, with_name=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.eval_set, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=self.num_workers)

        