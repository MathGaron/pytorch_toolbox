from torch.utils.data.dataset import Dataset
from abc import ABCMeta, abstractmethod


class LoaderBase(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, root, transform=None, target_transform=None):
        imgs = self.make_dataset(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def make_dataset(self, dir):
        """
        Should return a list of (data, target)
        :param path:
        :param class_to_idx:
        :return:
        """
        pass

    @abstractmethod
    def from_index(self, index):
        """
        should return a tuple : (data, target)
        :param index:
        :return:
        """
        pass

    def __getitem__(self, index):
        data, target = self.from_index(index)
        if not isinstance(data, list):
            data = [data]

        if self.transform is not None:
            for i in range(len(data)):
                data[i] = self.transform[i](data[i])
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.imgs)
