import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .tiny_imagenet_utils import download_and_extract_tiny_imagenet, generate_tiny_imagenet_data


class TinyImageNet(Dataset):
    def __init__(self, data_dir, transform, split_type, task_idx=None):
        self.df = self.load_tiny_imagenet_(data_dir, split_type)
        self.transform = transform
        self.task_id = task_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path, label = self.df.values[idx]

        img = Image.open(img_path)
        if img.mode == "L":
            img = img.convert("RGB")
        img = self.transform(img)

        if self.task_id is None:
            return img, label
        else:
            return img, label, self.task_id

    @staticmethod
    def load_tiny_imagenet_(data_dir, split_type):
        if not os.path.exists(data_dir):
            data_dir_ = os.path.dirname(data_dir)

            download_and_extract_tiny_imagenet(data_dir_)
            tiny_imagenet_data_dir = data_dir
            generate_tiny_imagenet_data(tiny_imagenet_data_dir)

        filename = split_type + "_tiny_imagenet200.csv"
        _path = os.path.join(data_dir, filename)
        df = pd.read_csv(_path)
        return df

    def filter_(self, categories):
        select_idx = (self.df["y_label"] == categories[0])
        for cat_ in categories[1:]:
            select_idx = select_idx | (self.df["y_label"] == cat_)

        self.df = self.df[select_idx]
        unique_cats = set(self.df["y_label"].unique().tolist())
        task_labels_2_idx_map = {label: idx for idx, label in enumerate(unique_cats)}
        self.df["y_label"] = self.df["y_label"].apply(lambda x: task_labels_2_idx_map[x])


train_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=63 / 255),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize(32),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# def get_tiny_imagenet_dataloaders(batch_size):
#     DATA_DIR = "./data"
#     TINY_IMAGENET_DATA_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")
#
#     train_dataset = TinyImageNet(TINY_IMAGENET_DATA_DIR, train_transforms, "train")
#     valid_dataset = TinyImageNet(TINY_IMAGENET_DATA_DIR, test_transforms, "valid")
#     test_dataset = TinyImageNet(TINY_IMAGENET_DATA_DIR, test_transforms, "test")
#
#     train_dataloader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
#     valid_dataloader = DataLoader(
#         valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
#     test_dataloader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
#
#     return train_dataloader, valid_dataloader, test_dataloader


def get_split_tiny_imagenet_dataloaders(batch_size, n_tasks):
    # classes_total = np.arange(200)
    # np.random.seed(5648968)
    # np.random.shuffle(classes_total)
    # task_classes = classes_total.reshape(-1, n_tasks).T
    task_classes = np.arange(200).reshape(-1, n_tasks).T
    n_classes_per_task = 200 // n_tasks

    def get_dataloader_(task_idx):
        # Retrieve train data
        DATA_DIR = "./data"
        TINY_IMAGENET_DATA_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")

        cur_task_classes = task_classes[task_idx]

        train_dataset = TinyImageNet(TINY_IMAGENET_DATA_DIR, train_transforms, "train", task_idx)
        valid_dataset = TinyImageNet(TINY_IMAGENET_DATA_DIR, test_transforms, "valid", task_idx)
        test_dataset = TinyImageNet(TINY_IMAGENET_DATA_DIR, test_transforms, "test", task_idx)
        train_dataset.filter_(cur_task_classes)
        valid_dataset.filter_(cur_task_classes)
        test_dataset.filter_(cur_task_classes)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        return train_dataloader, valid_dataloader, test_dataloader

    return get_dataloader_, n_classes_per_task


if __name__ == '__main__':
    DATA_DIR = "./data"
    TINY_IMAGENET_DATA_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")
    TinyImageNet.load_tiny_imagenet_(TINY_IMAGENET_DATA_DIR, "train")
