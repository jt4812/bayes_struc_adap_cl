import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST


class CLMNISTDatset(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, task_id: int = None,
                 knowledge_distillation=False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.has_task_id = task_id is not None
        self.task_id = task_id
        self.knowledge_distillation = knowledge_distillation

    def update_task_ids(self):
        self.task_ids = np.ones(len(self.data)) * self.task_id

    def one_hot_encode_targets(self):
        self.targets = F.one_hot(torch.Tensor(self.targets).long(), num_classes=len(np.unique(self.targets)))

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.data[idx], self.targets[idx]

        img = (img / 255.).ravel()

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.has_task_id:
            return img, target, self.task_ids[idx]
        else:
            return img, target

    def permute(self, permutation):
        self.data = self.data.reshape(-1, 784)[:, permutation].reshape(-1, 28, 28)

    def add_replay_data(self, img_list, labels, task_idx_list=None):
        img_list = torch.Tensor(img_list).type(torch.uint8).view(-1, 28, 28)
        self.data = torch.concatenate([self.data, img_list])
        self.targets = np.concatenate([self.targets, labels])

        assert (self.has_task_id and (task_idx_list is not None)) or (
                (not self.has_task_id) and (task_idx_list is None)), "Task id not stored by current dataset"
        if task_idx_list is not None:
            self.task_ids = np.concatenate([self.task_ids, task_idx_list])


class FashionMNISTDataset(FashionMNIST):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, task_id: int = None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.has_task_id = task_id is not None
        self.prev_task_id = task_id

    def update_task_ids(self):
        self.task_ids = np.ones(len(self.data)) * self.prev_task_id

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.data[idx], int(self.targets[idx])

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.has_task_id:
            return img, target, self.task_ids[idx]
        else:
            return img, target

    def add_replay_data(self, img_list, labels, task_idx_list=None):
        img_list = torch.Tensor(img_list).type(torch.uint8).view(-1, 28, 28)
        self.data = torch.concatenate([self.data, img_list])
        self.targets = np.concatenate([self.targets, labels])

        assert (self.has_task_id and (task_idx_list is not None)) or (
                (not self.has_task_id) and (task_idx_list is None)), "Task id not stored by current dataset"
        if task_idx_list is not None:
            self.task_ids = np.concatenate([self.task_ids, task_idx_list])


class MNISTFashionMNISTDataset(FashionMNIST):
    """ Merges MNIST and FashionMNIST datasets into one dataset. """

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, task_id: int = None) -> None:
        super().__init__(root, train, transform, target_transform, download)

        mnist_data = MNIST(root, train, transform, target_transform, download)
        self.data = torch.vstack([mnist_data.data, self.data])
        self.targets = torch.concat([mnist_data.targets, self.targets + 10])

        self.has_task_id = task_id is not None
        self.prev_task_id = task_id

    def update_task_ids(self):
        self.task_ids = np.ones(len(self.data)) * self.prev_task_id

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.data[idx], int(self.targets[idx])

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.has_task_id:
            return img, target, self.task_ids[idx]
        else:
            return img, target

    def add_replay_data(self, img_list, labels, task_idx_list=None):
        img_list = torch.Tensor(img_list).type(torch.uint8).view(-1, 28, 28)
        self.data = torch.concatenate([self.data, img_list])
        self.targets = np.concatenate([self.targets, labels])

        assert (self.has_task_id and (task_idx_list is not None)) or (
                (not self.has_task_id) and (task_idx_list is None)), "Task id not stored by current dataset"
        if task_idx_list is not None:
            self.task_ids = np.concatenate([self.task_ids, task_idx_list])


class CIFAR10Dataset(CIFAR10):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, task_id: int = None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.has_task_id = task_id is not None
        self.prev_task_id = task_id

    def update_task_ids(self):
        self.task_ids = np.ones(len(self.data)) * self.prev_task_id

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.data[idx], int(self.targets[idx])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.has_task_id:
            return img, target, self.task_ids[idx]
        else:
            return img, target

    def add_replay_data(self, img_list, labels, task_idx_list=None):
        img_list = torch.Tensor(img_list).type(torch.uint8).view(-1, 28, 28)
        self.data = torch.concatenate([self.data, img_list])
        self.targets = np.concatenate([self.targets, labels])

        assert (self.has_task_id and (task_idx_list is not None)) or (
                (not self.has_task_id) and (task_idx_list is None)), "Task id not stored by current dataset"
        if task_idx_list is not None:
            self.task_ids = np.concatenate([self.task_ids, task_idx_list])


class CIFAR100Dataset(CIFAR100):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, task_id: int = None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.has_task_id = task_id is not None
        self.prev_task_id = task_id

    def update_task_ids(self):
        self.task_ids = np.ones(len(self.data)) * self.prev_task_id

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.data[idx], int(self.targets[idx])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.has_task_id:
            return img, target, self.task_ids[idx]
        else:
            return img, target

    def add_replay_data(self, img_list, labels, task_idx_list=None):
        img_list = torch.Tensor(img_list).type(torch.uint8).view(-1, 28, 28)
        self.data = torch.concatenate([self.data, img_list])
        self.targets = np.concatenate([self.targets, labels])

        assert (self.has_task_id and (task_idx_list is not None)) or (
                (not self.has_task_id) and (task_idx_list is None)), "Task id not stored by current dataset"
        if task_idx_list is not None:
            self.task_ids = np.concatenate([self.task_ids, task_idx_list])


dataset_mapper = {
    "split_mnist": {
        "dataset": CLMNISTDatset,
        "n_tasks": 5,
        "n_classes_per_task": 2,
        "mu": None,
        "std": None
    },
    "split_fashion_mnist": {
        "dataset": FashionMNISTDataset,
        "n_tasks": 5,
        "n_classes_per_task": 2,
        "mu": None,
        "std": None
    },
    "split_mnist_fashion_mnist": {
        "dataset": MNISTFashionMNISTDataset,
        "n_tasks": 10,
        "n_classes_per_task": 2,
        "mu": None,
        "std": None
    },
    "one_mnist": {
        "dataset": CLMNISTDatset,
        "n_tasks": 10,
        "n_classes_per_task": 1,
        "mu": None,
        "std": None
    },
    "one_fashion_mnist": {
        "dataset": FashionMNISTDataset,
        "n_tasks": 10,
        "n_classes_per_task": 1,
        "mu": None,
        "std": None
    },
    "split_cifar10": {
        "dataset": CIFAR10Dataset,
        "n_tasks": 5,
        "n_classes_per_task": 2,
        "mu": (0.49139968, 0.48215827, 0.44653124),
        "std": (0.24703233, 0.24348505, 0.26158768)
    },
    "split_cifar100_5": {
        "dataset": CIFAR100Dataset,
        "n_tasks": 5,
        "n_classes_per_task": 20,
        "mu": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "split_cifar100_10": {
        "dataset": CIFAR100Dataset,
        "n_tasks": 10,
        "n_classes_per_task": 10,
        "mu": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "split_cifar100_20": {
        "dataset": CIFAR100Dataset,
        "n_tasks": 20,
        "n_classes_per_task": 5,
        "mu": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "split_cifar100_50": {
        "dataset": CIFAR100Dataset,
        "n_tasks": 50,
        "n_classes_per_task": 2,
        "mu": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


def get_perm_mnist_dataloaders(batch_size, cl_type="task_il", n_tasks=5, seed=100, get_dataloader=True,
                               use_distillation=False, use_validation_dataset=False):
    np.random.seed(seed)
    permutations = [np.random.permutation(784) for _ in range(n_tasks)]

    def get_dataloader_(task_idx):
        assert task_idx < n_tasks

        idx_permute = permutations[task_idx]
        transform_ = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

        train_dataset = CLMNISTDatset(root='./data', train=True, transform=transform_, download=True, task_id=task_idx)
        train_dataset.permute(idx_permute)
        train_dataset.update_task_ids()

        test_dataset = CLMNISTDatset(root='./data', train=False, transform=transform_)
        test_dataset.permute(idx_permute)

        if use_distillation:
            train_dataset.one_hot_encode_targets()
            test_dataset.one_hot_encode_targets()

        if not get_dataloader:
            return train_dataset, test_dataset

        if use_validation_dataset:
            len_valid_dataset = int(0.1 * len(train_dataset))
            train_dataset, valid_dataset = torch.utils.data.random_split(
                train_dataset, [len(train_dataset) - len_valid_dataset, len_valid_dataset], generator=torch.Generator().manual_seed(42))
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                      shuffle=False)
        else:
            valid_dataloader = None

        train_dataloader_ = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                       shuffle=True)
        test_dataloader_ = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                      shuffle=False)
        return train_dataloader_, valid_dataloader, test_dataloader_

    return get_dataloader_, 10


def get_transform(dataset_, input_as_image):
    transform_ls = [transforms.ToTensor()]

    # TODO: dataset align not implemented for other transformations in other datasets
    if dataset_["mu"] is not None:
        transform_ls.append(transforms.Normalize(dataset_["mu"], dataset_["std"]))

    if not input_as_image:
        transform_ls.append(transforms.Lambda(lambda x: x if input_as_image else x.view(-1)))

    transform_ = transforms.Compose(transform_ls)

    return transform_


def get_split_dataloaders(batch_size, single_head, cl_type, dataset="mnist", get_dataloader=True, input_as_image=False,
                          test_batch_size=None, use_distillation=False, use_validation_dataset=False):
    assert cl_type in ["task_il", "domain_il", "class_il"]

    dataset_ = dataset_mapper[dataset]
    n_tasks = dataset_["n_tasks"]
    n_classes_per_task = dataset_["n_classes_per_task"]
    cat_splits = np.arange(n_tasks * n_classes_per_task).reshape(n_tasks, -1)

    if "split_cifar100" in dataset:  # bucketing == "across":
        cat_splits = np.arange(n_tasks * n_classes_per_task).reshape(-1, n_tasks).T

    transform_ = get_transform(dataset_, input_as_image)

    def get_dataloader_(task_idx):

        task_categories = cat_splits[task_idx]
        train_dataset = dataset_["dataset"](
            root='./data', train=True, transform=transform_, download=True, task_id=task_idx)
        train_dataset.targets = np.array(train_dataset.targets)

        train_idx = (train_dataset.targets == task_categories[0])

        if len(task_categories) > 1:
            for task_cat in task_categories[1:]:
                train_idx = train_idx | (train_dataset.targets == task_cat)

        train_dataset.data, train_dataset.targets = train_dataset.data[train_idx], train_dataset.targets[train_idx]
        train_dataset.update_task_ids()

        test_dataset = dataset_["dataset"](root='./data', train=False, transform=transform_)
        test_dataset.targets = np.array(test_dataset.targets)

        test_idx = (test_dataset.targets == task_categories[0])
        if len(task_categories) > 1:
            for task_cat in task_categories[1:]:
                test_idx = test_idx | (test_dataset.targets == task_cat)

        test_dataset.data, test_dataset.targets = test_dataset.data[test_idx], test_dataset.targets[test_idx]

        if (not single_head) or (cl_type == "domain_il"):
            # Align targets for the datasets to match multi-head and domain incremental settings
            task_labels_2_idx_map = {label: idx for idx, label in enumerate(np.unique(train_dataset.targets).tolist())}

            train_dataset.targets = np.array([task_labels_2_idx_map[label] for label in train_dataset.targets])
            test_dataset.targets = np.array([task_labels_2_idx_map[label] for label in test_dataset.targets])

        if use_distillation:
            train_dataset.one_hot_encode_targets()
            test_dataset.one_hot_encode_targets()

        if use_validation_dataset:
            len_valid_dataset = int(0.1 * len(train_dataset))
            train_dataset, valid_dataset = torch.utils.data.random_split(
                train_dataset, [len(train_dataset) - len_valid_dataset, len_valid_dataset], generator=torch.Generator().manual_seed(42))
            valid_dataloader = create_dataloader_instance(valid_dataset, batch_size, shuffle=False)
        else:
            valid_dataloader = None
        train_dataloader_ = create_dataloader_instance(train_dataset, batch_size, shuffle=True)
        test_dataloader_ = create_dataloader_instance(test_dataset, test_batch_size if test_batch_size else batch_size,
                                                      shuffle=False)

        return train_dataloader_, valid_dataloader, test_dataloader_

    return get_dataloader_, n_classes_per_task


def get_not_mnist_dataloaders(batch_size, single_head, cl_type, dataset="mnist", get_dataloader=True,
                              input_as_image=False, use_distillation=False):
    assert cl_type in ["task_il", "domain_il", "class_il"]

    notmnist_mat_path = os.path.join("data", "notMNIST_small.mat")
    out = loadmat(notmnist_mat_path)
    X_all = out['images'].transpose(2, 0, 1) / 255.
    Y_all = out['labels'].squeeze()
    cat_splits = np.arange(10).reshape(-1, 1)
    seed = 0

    def get_dataloader_(task_idx):
        task_categories = cat_splits[task_idx]
        ind = [np.where(Y_all == i)[0] for i in task_categories]
        ind = np.concatenate(ind)

        X = X_all[ind]
        X = X.reshape(X.shape[0], -1)
        Y = Y_all[ind]

        # np.random.seed(seed)
        # N_train = int(X.shape[0] * 0.8)
        # ind = np.random.permutation(range(X.shape[0]))
        # X_train = torch.tensor(X[ind[:N_train]])
        # Y_train = torch.tensor(Y[ind[:N_train]])
        # X_test = torch.tensor(X[ind[N_train:]])
        # Y_test = torch.tensor(Y[ind[N_train:]])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        X_train = torch.tensor(X_train)
        Y_train = torch.tensor(Y_train)
        X_test = torch.tensor(X_test)
        Y_test = torch.tensor(Y_test)

        # import pdb; pdb.set_trace()

        train_dataset = TensorDataset(X_train, Y_train, torch.ones_like(Y_train) * task_idx)
        test_dataset = TensorDataset(X_test, Y_test, torch.ones_like(Y_test) * task_idx)

        if not get_dataloader:
            return train_dataset, test_dataset

        train_dataloader_ = create_dataloader_instance(train_dataset, batch_size, shuffle=True)
        test_dataloader_ = create_dataloader_instance(test_dataset, batch_size, shuffle=False)

        return train_dataloader_, None, test_dataloader_

    return get_dataloader_, 1


def get_split_not_mnist_dataloaders(batch_size, single_head, cl_type, dataset="mnist", get_dataloader=True,
                                    input_as_image=False, use_distillation=False):
    assert cl_type in ["task_il", "domain_il", "class_il"]

    notmnist_mat_path = os.path.join("data", "notMNIST_small.mat")
    out = loadmat(notmnist_mat_path)
    X_all = out['images'].transpose(2, 0, 1) / 255.
    Y_all = out['labels'].squeeze()
    cat_splits = np.arange(10).reshape(-1, 2)
    seed = 0

    def get_dataloader_(task_idx):
        task_categories = cat_splits[task_idx]

        ind = [np.where(Y_all == i)[0] for i in task_categories]

        ind = np.concatenate(ind)

        X = X_all[ind]
        Y = Y_all[ind]

        np.random.seed(seed)
        N_train = int(X.shape[0] * 0.8)
        ind = np.random.permutation(range(X.shape[0]))
        X = X.reshape(X.shape[0], -1)
        X_train = torch.tensor(X[ind[:N_train]])
        Y_train = torch.tensor(Y[ind[:N_train]]) - cat_splits.shape[-1] * task_idx

        X_test = torch.tensor(X[ind[N_train:]])
        Y_test = torch.tensor(Y[ind[N_train:]]) - cat_splits.shape[-1] * task_idx

        train_dataset = TensorDataset(X_train, Y_train, torch.ones_like(Y_train) * task_idx)
        test_dataset = TensorDataset(X_test, Y_test, torch.ones_like(Y_test) * task_idx)

        if not get_dataloader:
            return train_dataset, test_dataset

        train_dataloader_ = create_dataloader_instance(train_dataset, batch_size, shuffle=True)
        test_dataloader_ = create_dataloader_instance(test_dataset, batch_size, shuffle=False)

        return train_dataloader_, None, test_dataloader_

    return get_dataloader_, 2


def create_dataloader_instance(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=shuffle)
    return dataloader
