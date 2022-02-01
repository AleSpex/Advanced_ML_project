from __future__ import print_function
import torch.utils.data as data
from PIL import Image



class Dataset(data.Dataset):

    def __init__(self, data, label,img_transformer=None):
        self.img_transformer = img_transformer
        self.data = data
        self.labels = label

    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]
        img = Image.open(img).convert('RGB')

        img = self.img_transformer(img)

        return img, target

    def __len__(self):
        return len(self.data)
