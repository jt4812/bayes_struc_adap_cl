import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MixCIFAR(Dataset):
    def __init__(self, data_dir, transform, train, task_idx=None):
        self.data, self.targets = self.get_dataset(train)
        self.transform = transform
        self.task_id = task_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.task_id is None:
            return img, label
        else:
            return img, label, self.task_id

    @staticmethod
    def get_dataset(is_train):
        n_cifar100_classes = 50
        selected_classes = np.arange(n_cifar100_classes)

        if is_train:
            cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
            cifar100_train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)

            cifar100_train_indices = np.isin(cifar100_train_dataset.targets, selected_classes)
            cifar100_data_train = cifar100_train_dataset.data[cifar100_train_indices]
            cifar100_targets_train = np.array(cifar100_train_dataset.targets)[cifar100_train_indices] + 10

            data = np.vstack([cifar10_train_dataset.data, cifar100_data_train])
            targets = np.concatenate([cifar10_train_dataset.targets, cifar100_targets_train])

        else:
            cifar10_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
            cifar100_test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)

            cifar100_test_indices = np.isin(cifar100_test_dataset.targets, selected_classes)
            cifar100_data_test = cifar100_test_dataset.data[cifar100_test_indices]
            cifar100_targets_test = np.array(cifar100_test_dataset.targets)[cifar100_test_indices] + 10

            data = np.vstack([cifar10_test_dataset.data, cifar100_data_test])
            targets = np.concatenate([cifar10_test_dataset.targets, cifar100_targets_test])

        return data, targets

    def filter_(self, categories):
        select_idx = (self.targets == categories[0])
        for cat_ in categories[1:]:
            select_idx = select_idx | (self.targets == cat_)

        self.data = self.data[select_idx]
        self.targets = self.targets[select_idx]

        task_labels_2_idx_map = {label: idx for idx, label in enumerate(np.unique(self.targets).tolist())}
        self.targets = np.array([task_labels_2_idx_map[x] for x in self.targets])


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24703233, 0.24348505, 0.26158768)),
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24703233, 0.24348505, 0.26158768)),
])


def get_split_cifarmix_dataloaders_(batch_size):
    train_dataset = MixCIFAR("./data", train_transforms, train=True)
    test_dataset = MixCIFAR("./data", test_transforms, train=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_dataloader, test_dataloader


# def get_split_cifarmix_dataloaders(batch_size, n_tasks=6):
def get_split_cifarmix_dataloaders(batch_size, use_validation_dataset=False):
    n_tasks = 6
    task_classes = np.arange(60).reshape(n_tasks, -1).T

    n_classes_per_task = 60 // n_tasks

    def get_dataloader_(task_idx):
        # Retrieve train data
        DATA_DIR = "./data"

        cur_task_classes = task_classes[task_idx]

        train_dataset = MixCIFAR(DATA_DIR, train_transforms, train=True, task_idx=task_idx)
        test_dataset = MixCIFAR(DATA_DIR, test_transforms, train=False, task_idx=task_idx)
        train_dataset.filter_(cur_task_classes)
        test_dataset.filter_(cur_task_classes)

        # train_dataloader = DataLoader(
        #     train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        # test_dataloader = DataLoader(
        #     test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

        if use_validation_dataset:
            len_valid_dataset = int(0.1 * len(train_dataset))
            train_dataset, valid_dataset = torch.utils.data.random_split(
                train_dataset, [len(train_dataset) - len_valid_dataset, len_valid_dataset],
                generator=torch.Generator().manual_seed(42))
            valid_dataloader = create_dataloader_instance(valid_dataset, batch_size, shuffle=False)
        else:
            valid_dataloader = None
        train_dataloader_ = create_dataloader_instance(train_dataset, batch_size, shuffle=True)
        test_dataloader_ = create_dataloader_instance(test_dataset, batch_size, shuffle=False)

        return train_dataloader_, valid_dataloader, test_dataloader_

    return get_dataloader_, n_classes_per_task


def create_dataloader_instance(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=shuffle)
    return dataloader
