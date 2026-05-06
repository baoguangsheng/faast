import os
import torch
import json
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, Omniglot
import datasets
from tqdm import tqdm
import numpy as np
from PIL import Image
from collections import defaultdict


class FewShotBatchSampler(object):
    def __init__(self, labels, num_shot, num_way=100, num_iter=600):
        super().__init__()
        self.labels = labels
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_iter = num_iter

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.num_shot
        cpi = self.num_way or len(self.classes)  # classes per iteration
        assert len(self.classes) >= cpi, f"Not enough classes {len(self.classes)} to sample {cpi} per batch."

        for it in range(self.num_iter):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                assert self.numel_per_class[label_idx] >= spc, f"Not enough samples {self.numel_per_class[label_idx]} to sample {spc} for class {c}."
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.num_iter


class MiniImageNetDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        """
        data_dict: output of mini-imagenet-cache-*.pkl, containing:
            - 'image_data': numpy array of shape (N, 84, 84, 3)
            - 'class_dict': {class_name: [indices]}
        """
        self.transform = transform
        self.images = data_dict['image_data']  # numpy array, uint8
        self.labels = np.zeros(len(self.images), dtype=np.int64)
        # Map class_name → integer label 0..C-1
        class_dict = data_dict['class_dict']
        self.classes = list(sorted(class_dict.keys()))   # ensure deterministic ordering
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        # Assign labels according to class_dict
        for cls_name, sample_indices in class_dict.items():
            cls_idx = self.class_to_idx[cls_name]
            self.labels[sample_indices] = cls_idx

    @classmethod
    def random_split_half(cls, data_dict, class_names, seed=42):
        images = data_dict['image_data']
        class_dict = data_dict['class_dict']
        # split each class's indices into train/test
        train_indices = []
        test_indices = []
        train_class_dict = defaultdict(list)
        test_class_dict = defaultdict(list)
        rng = np.random.RandomState(seed)
        for cls_name, sample_indices in class_dict.items():
            rng.shuffle(sample_indices)
            half_size = len(sample_indices) // 2
            assert len(sample_indices) % 2 == 0, "Class with odd number of samples found."
            if class_names is not None:
                cls_name = class_names[cls_name]
            train_class_dict[cls_name] = (len(train_indices) + np.arange(half_size)).tolist()
            test_class_dict[cls_name] = (len(test_indices) + np.arange(half_size)).tolist()
            train_indices.extend(sample_indices[:half_size])
            test_indices.extend(sample_indices[half_size:])
        # create train/test subsets
        train_data_dict = {'image_data': images[train_indices],
                             'class_dict': train_class_dict}
        test_data_dict = {'image_data': images[test_indices],
                            'class_dict': test_class_dict}
        return train_data_dict, test_data_dict
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]               # HWC (84, 84, 3), uint8
        label = int(self.labels[idx])

        # Convert numpy to PIL for transforms
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def prepare_labels(dataset, split, cache_path):
    label_file = f'{cache_path}/labels_{split}.txt'
    if os.path.exists(label_file):
        print(f'Loading cached labels from {label_file}...')
        labels = json.load(open(label_file, 'r'))
    else:
        print(f'Saving labels to {label_file}...')
        labels = [label for _, label in dataset]
        json.dump(labels, open(label_file, 'w'))
    setattr(dataset, 'labels', labels)
    return labels


def load_cifar10(data_path, transform=None):
    print('Loading CIFAR-10 dataset from:', data_path)
    supportset = CIFAR10(root=data_path, train=True, download=False, transform=transform)
    queryset = CIFAR10(root=data_path, train=False, download=False, transform=transform)
    data_path = data_path + '/cifar-10-batches-py'
    prepare_labels(supportset, 'support', data_path)
    prepare_labels(queryset, 'query', data_path)    
    return supportset, queryset


def _load_mini_imagenet(data_path, transform=None, use_class_id=False):
    # load pickle files from https://github.com/renmengye/few-shot-ssl-public#miniimagenet
    # load class mapping file from https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
    # dict: image_data[84*84*3], class_dict[str: int]
    import pickle
    import json
    testset = pickle.load(open(data_path + '/mini-imagenet-cache-test.pkl', 'rb'))
    class_names = None if use_class_id else json.load(open(data_path + '/class_names.json', 'r'))
    supportset, queryset = MiniImageNetDataset.random_split_half(testset, class_names, seed=42)
    supportset = MiniImageNetDataset(supportset, transform=transform)
    queryset = MiniImageNetDataset(queryset, transform=transform)
    return supportset, queryset


def load_mini_imagenet(data_path, transform=None):
    data_path = data_path + '/mini-imagenet'
    supportset, queryset = _load_mini_imagenet(data_path, transform)
    prepare_labels(supportset, 'support', data_path)
    prepare_labels(queryset, 'query', data_path)
    return supportset, queryset


def load_mini_imagenet_id(data_path, transform=None):
    data_path = data_path + '/mini-imagenet'
    supportset, queryset = _load_mini_imagenet(data_path, transform, use_class_id=True)
    prepare_labels(supportset, 'support-id', data_path)
    prepare_labels(queryset, 'query-id', data_path)
    return supportset, queryset


dataset_fns = {
    'mini-imagenet': load_mini_imagenet,
    'mini-imagenet-id': load_mini_imagenet_id,
    'cifar10': load_cifar10,
}


if __name__ == "__main__":

    trainset, testset = load_cifar10('./data/vision')
    for i in tqdm(range(len(trainset)), desc="Verifying trainset"):
        assert len(trainset[i][0].shape) == 3
    for i in tqdm(range(len(testset)), desc="Verifying testset"):
        assert len(testset[i][0].shape) == 3
    print(trainset, testset)
    print('testset:', testset[0])
    # print('Classes:', testset.classes)
    print('Image:', trainset[0][0].shape)