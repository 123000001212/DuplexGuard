"""Super-classes of common datasets to extract id information per image."""
import torch
import torchvision
import PIL
import numpy as np
import pathlib
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import csv
from typing import Any, Callable, Optional, Tuple
from ..consts import *   # import all mean/std constants

import torchvision.transforms as transforms
from PIL import Image
import os
import glob

from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets import ImageFolder

# Block ImageNet corrupt EXIF warnings
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def construct_datasets(dataset, data_path, normalize=True, train_indices=None):
    """Construct datasets with appropriate transforms."""
    # Compute mean, std:
    if dataset == 'CIFAR100':
        trainset = CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if cifar100_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar100_mean, cifar100_std
    elif dataset == 'CIFAR10':
        trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
    elif dataset == 'MNIST':
        trainset = MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if mnist_mean is None:
            cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
            data_mean = (torch.mean(cc, dim=0).item(),)
            data_std = (torch.std(cc, dim=0).item(),)
        else:
            data_mean, data_std = mnist_mean, mnist_std
    elif dataset == 'SVHN':
        trainset = SVHN(root=data_path, split='train', download=False, transform=transforms.ToTensor())
        cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()
        trainset.classes=[str(i) for i in range(10)]
    elif dataset == 'GTSRB':
        trainset = GTSRB(root=data_path, split='train', download=False, transform=transforms.ToTensor())
        cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()
        trainset.classes=[str(i) for i in range(43)]
    elif dataset == 'STL10':
        trainset = STL10(root=data_path, split='train', download=True, transform=transforms.ToTensor())
        cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()
        trainset.classes=[str(i) for i in range(10)]
    elif dataset == 'ImageNet':
        trainset = ImageNet(root=data_path, split='train', download=False, transform=transforms.ToTensor())
        if imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = imagenet_mean, imagenet_std
    elif dataset == 'ImageNet1k':
        trainset = ImageNet1k(root=data_path, split='train', download=False, transform=transforms.ToTensor())
        if imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = imagenet_mean, imagenet_std
    elif dataset == 'TinyImageNet':
        trainset = TinyImageNet(root=data_path, split='train', transform=transforms.ToTensor())
        if tiny_imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = tiny_imagenet_mean, tiny_imagenet_std

    elif dataset == 'ImageNette':
        trainset = ImageNette(root=data_path, split='train', transform=transforms.ToTensor())
        cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()

    else:
        raise ValueError(f'Invalid dataset {dataset} given.')

    if normalize:
        print(f'Data mean is {data_mean}, \nData std  is {data_std}.')
        trainset.data_mean = data_mean
        trainset.data_std = data_std
    else:
        print('Normalization disabled.')
        trainset.data_mean = (0.0, 0.0, 0.0)
        trainset.data_std = (1.0, 1.0, 1.0)

    # Setup data
    if dataset in ['ImageNet', 'ImageNet1k']:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    if dataset=='ImageNette':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform_train

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    if dataset == 'CIFAR100':
        validset = CIFAR100(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR10':
        validset = CIFAR10(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'MNIST':
        validset = MNIST(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'SVHN':
        validset = SVHN(root=data_path, split='test', download=False, transform=transform_valid)
    elif dataset == 'GTSRB':
        validset = GTSRB(root=data_path, split='test', download=False, transform=transform_valid)
    elif dataset == 'STL10':
        validset = STL10(root=data_path, split='test', download=False, transform=transform_valid)
    elif dataset == 'TinyImageNet':
        validset = TinyImageNet(root=data_path, split='val', transform=transform_valid)
    elif dataset == 'ImageNet':
        # Prepare ImageNet beforehand in a different script!
        # We are not going to redownload on every instance
        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        validset = ImageNet(root=data_path, split='val', download=False, transform=transform_valid)
    elif dataset == 'ImageNet1k':
        # Prepare ImageNet beforehand in a different script!
        # We are not going to redownload on every instance
        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        validset = ImageNet1k(root=data_path, split='val', download=False, transform=transform_valid)
    elif dataset == 'ImageNette':
        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        validset = ImageNette(root=data_path, split='val', transform=transform_valid)

    if normalize:
        validset.data_mean = data_mean
        validset.data_std = data_std
    else:
        validset.data_mean = (0.0, 0.0, 0.0)
        validset.data_std = (1.0, 1.0, 1.0)

    if train_indices is not None:
        if dataset=='ImageNette':
            trainset.dataset.imgs=[trainset.dataset.imgs[i] for i in train_indices]
            trainset.dataset.samples=[trainset.dataset.samples[i] for i in train_indices]
            trainset.dataset.targets=[trainset.dataset.targets[i] for i in train_indices]

            # trainset.dataset.samples=np.array(trainset.dataset.samples)[train_indices].tolist()
            

        elif dataset=='CIFAR10':
            trainset.data=trainset.data[train_indices]
            trainset.targets=np.array(trainset.targets)[train_indices].tolist()

        elif dataset=='STL10' or dataset=='SVHN' or dataset=='GTSRB':
            trainset.data=trainset.data[train_indices]
            trainset.labels=np.array(trainset.labels)[train_indices].tolist()
        else:
            raise('Dataset not supported.')


    return trainset, validset


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)


class Deltaset(torch.utils.data.Dataset):
    def __init__(self, dataset, delta):
        self.dataset = dataset
        self.delta = delta

    def __getitem__(self, idx):
        (img, target, index) = self.dataset[idx]
        return (img + self.delta[idx], target, index)

    def __len__(self):
        return len(self.dataset)


class CIFAR10(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

class SVHN(torchvision.datasets.SVHN):
    """Super-class SVHN to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.transpose(1,2,0)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.labels[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

class STL10(torchvision.datasets.STL10):
    """Super-class STL10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img.transpose(1,2,0)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.labels[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class CIFAR100(torchvision.datasets.CIFAR100):
    """Super-class CIFAR100 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class MNIST(torchvision.datasets.MNIST):
    """Super-class MNIST to return image ids with images."""

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class ImageNet(torchvision.datasets.ImageNet):
    """Overwrite torchvision ImageNet to change metafile location if metafile cannot be written due to some reason."""

    def __init__(self, root, split='train', download=False, **kwargs):
        """Use as torchvision.datasets.ImageNet."""
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        try:
            wnid_to_classes = load_meta_file(self.root)[0]
        except RuntimeError:
            torchvision.datasets.imagenet.META_FILE = os.path.join(os.path.expanduser('~/data/'), 'meta.bin')
            try:
                wnid_to_classes = load_meta_file(self.root)[0]
            except RuntimeError:
                self.parse_archives()
                wnid_to_classes = load_meta_file(self.root)[0]

        torchvision.datasets.ImageFolder.__init__(self, self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        """Scrub class names to be a single string."""
        scrubbed_names = []
        for name in self.classes:
            if isinstance(name, tuple):
                scrubbed_names.append(name[0])
            else:
                scrubbed_names.append(name)
        self.classes = scrubbed_names

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, idx) where target is class_index of the target class.

        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        _, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index



class ImageNet1k(ImageNet):
    """Overwrite torchvision ImageNet to limit it to less than 1mio examples.

    [limit/per class, due to automl restrictions].
    """

    def __init__(self, root, split='train', download=False, limit=950, **kwargs):
        """As torchvision.datasets.ImageNet except for additional keyword 'limit'."""
        super().__init__(root, split, download, **kwargs)

        # Dictionary, mapping ImageNet1k ids to ImageNet ids:
        self.full_imagenet_id = dict()
        # Remove samples above limit.
        examples_per_class = torch.zeros(len(self.classes))
        new_samples = []
        new_idx = 0
        for full_idx, (path, target) in enumerate(self.samples):
            if examples_per_class[target] < limit:
                examples_per_class[target] += 1
                item = path, target
                new_samples.append(item)
                self.full_imagenet_id[new_idx] = full_idx
                new_idx += 1
            else:
                pass
        self.samples = new_samples
        print(f'Size of {self.split} dataset reduced to {len(self.samples)}.')




"""
    The following class is heavily based on code by Meng Lee, mnicnc404. Date: 2018/06/04
    via
    https://github.com/leemengtaiwan/tiny-imagenet/blob/master/TinyImageNet.py
"""


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Author: Meng Lee, mnicnc404
    Date: 2018/06/04
    References:
        - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """

    EXTENSION = 'JPEG'
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = 'wnids.txt'
    VAL_ANNOTATION_FILE = 'val_annotations.txt'
    CLASSES = 'words.txt'

    def __init__(self, root, split='train', transform=None, target_transform=None):
        """Init with split, transform, target_transform. use --cached_dataset data is to be kept in memory."""
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping

        # build class label - number mapping
        with open(os.path.join(self.root, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(root, self.CLASSES), 'r') as file:
            for line in file:
                label_text, word = line.split('\t')
                label_text_to_word[label_text] = word.split(',')[0].rstrip('\n')
        self.classes = [label_text_to_word[label] for label in self.label_texts]

        # Prepare index - label mapping
        self.targets = [self.labels[os.path.basename(file_path)] for file_path in self.image_paths]

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return a triplet of image, label, index."""
        file_path, target = self.image_paths[index], self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = Image.open(file_path)
        img = img.convert("RGB")
        img = self.transform(img) if self.transform else img
        if self.split == 'test':
            return img, None, index
        else:
            return img, target, index


    def get_target(self, index):
        """Return only the target and its id."""
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

class ImageNette(torch.utils.data.Dataset):
    def __init__(self, root, split='train',transform=None):
        self.target_transform=None
        self.transform=transform
        self.dataset=ImageFolder(root=root+'/'+split,transform=None)

        self.classes=self.dataset.classes
        self.classes_to_idx=self.dataset.class_to_idx

        self.poison_indices=[]
        self.poison_delta=[]
    
    def __len__(self):
        """Return length via image paths."""
        return len(self.dataset)

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, idx) where target is class_index of the target class.

        """
        #print(f'Index={index}')
        if int(index) in self.poison_indices:
            #print(f'Adding poison on {index}')
            sample, target = self.dataset[int(index)]
            sample=self.transform(sample)
            sample = sample + self.poison_delta[self.indexof(int(index),self.poison_indices)]
        else:
            sample, target = self.dataset[index]
            sample=self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
    
    def get_target(self, index):
        """Return only the target and its id."""
        _, target = self.dataset[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

    def indexof(self,I,L):
        for i in range(len(L)):
            if L[i]==I:
                return i
        else:
            raise('index not found')
        
class GTSRB(torchvision.datasets.VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "gtsrb"
        self._target_folder = (
            self._base_folder / "GTSRB" / ("Training" if self._split == "train" else "Final_Test/Images")
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self._split == "train":
            samples = make_dataset(str(self._target_folder), extensions=(".ppm",))
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform
        init_transform = transforms.Compose([transforms.Resize((32,32))])
        datas=[(PIL.Image.open(path).convert("RGB"),target) for path,target in samples]
        self.data=np.array([np.array(init_transform(img)) for img,label in datas]) # 12630,32,32,3
        self.labels=np.array([label for img,label in datas]) # 12630,

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")
        """
        sample = self.data[index]
        target = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()

    def download(self) -> None:
        if self._check_exists():
            return

        base_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

        if self._split == "train":
            download_and_extract_archive(
                f"{base_url}GTSRB-Training_fixed.zip",
                download_root=str(self._base_folder),
                md5="513f3c79a4c5141765e10e952eaa2478",
            )
        else:
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_Images.zip",
                download_root=str(self._base_folder),
                md5="c7e4e6327067d32654124b0fe9e82185",
            )
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_GT.zip",
                download_root=str(self._base_folder),
                md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            )
    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.labels[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
