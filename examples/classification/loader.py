import os
import numpy as np
from PIL import Image

from pytorch_toolbox.loader.loader_base import LoaderBase


class CatVsDogLoader(LoaderBase):
    def __init__(self, root, transform=[], target_transform=[]):
        classes, class_to_idx = self.find_classes(root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        super().__init__(root, transform, target_transform)

    @staticmethod
    def find_classes(dir):
        """
        Returns a list of classes : check all files and take the first part separated by a "."
        ex : cat.4.jpg : class = cat
        :param dir:
        :return:
        """
        classes = [d.split(".")[0] for d in os.listdir(dir)]
        classes = list(set(classes))
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir):
        images = []
        dir = os.path.expanduser(dir)
        files = [x for x in os.listdir(dir)]
        for file in files:
            path = os.path.join(dir, file)
            label = self.class_to_idx[file.split(".")[0]]
            images.append((path, label))
        return images

    def from_index(self, index):
        rgb_path, label = self.imgs[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        return rgb_img, label
