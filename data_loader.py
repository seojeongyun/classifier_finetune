import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

cudnn.benchmark = True
plt.ion()   # interactive mode

class dataloader:
    def __init__(self):
        self.path = '/storage/jysuh/hymenoptera_data'
        self.transform = self.data_transfrom()
        self.image_datasets = self.get_datasets()
        self.dataloaders = self.get_loaders()
        self.len = self.get_len()
        self.cls_name = self.get_clsname()

    def data_transfrom(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return data_transforms

    def get_datasets(self):
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.path, x),self.transform[x]) for x in ['train', 'val']}
        return image_datasets

    def get_loaders(self):
        dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x],
                                                      batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
        return dataloaders

    def get_len(self):
        dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        return dataset_sizes

    def get_clsname(self):
        class_names = self.image_datasets['train'].classes
        return class_names
