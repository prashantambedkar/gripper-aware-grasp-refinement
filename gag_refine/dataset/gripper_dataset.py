import os

from torch.utils.data import Dataset
import numpy as np


class GripperPointCloudData(Dataset):
    def __init__(self, dataset_dir, split='train', noise_std=None):
        self.dataset_dir = dataset_dir
        self.split = split
        self.noise_std = noise_std

        self.file_indices = self.load_split()

    def load_split(self):
        split_file = os.path.join(self.dataset_dir, f'{self.split}.lst')
        with open(split_file, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        return lines

    def apply_noise(self, points):
        noise = self.noise_std * np.random.randn(*points.shape)
        points = points + noise
        return points

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, index):
        item_fn = os.path.join(self.dataset_dir, f'{self.file_indices[index]}.npz')
        data = dict(np.load(item_fn))
        if self.noise_std is not None and self.noise_std != 0:
            data['points'] = self.apply_noise(data['points'])
        return data
