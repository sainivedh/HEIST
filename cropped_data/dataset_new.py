import numpy as np
import albumentations
import torch
from torch.utils.data import Dataset
import random

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as transforms
from torch.autograd import Variable
import data

import config

class ClassificationDataset(Dataset):
    def __init__(self,image_paths, targets):
        super(ClassificationDataset, self).__init__()
        self.image_paths = image_paths
        self.targets = targets
        self.shuffle_indices = list(range(len(self.image_paths)))
        self.shuffle_start_index = 0


        self.transformer = data.resizeNormalize((100,32))



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("L")
        targets = self.targets[item]
        image = self.transformer(image)
        image = image.view(1, *image.size())
        image = Variable(image)

        return {
            "images": image,
            "targets": torch.tensor(targets, dtype=torch.long),
        }

    def get_batch_indices(self, batch_size):
        end = self.shuffle_start_index + batch_size
        batch = []
        if end >= len(self.image_paths):
            batch = [i for i in self.shuffle_indices[0:end%len(self.image_paths)]]
            end = len(self.image_paths)
        return batch + [i for i in self.shuffle_indices[self.shuffle_start_index:end]]

    def get_image_target_batch(self, batch_indices):
        images_batch = []
        targets_batch = []
        for i in batch_indices:
            sample = self.__getitem__(i)
            images_batch.append(sample["images"])
            targets_batch.append(sample["targets"])
        return images_batch, targets_batch

    def get_batch(self, batch_size):
        batch_indices = self.get_batch_indices(batch_size)
        images_batch, targets_batch = self.get_image_target_batch(batch_indices)

        self.shuffle_start_index += batch_size + 1

        if self.shuffle_start_index >= len(self.image_paths):
            random.shuffle(self.shuffle_indices)
            self.shuffle_start_index = 0

        return images_batch, targets_batch


#data = ClassificationDataset(images_path, outputs, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
#print(data.__getitem__(np.random.randint(1600,size=(1,1)).squeeze())["images"].shape)

