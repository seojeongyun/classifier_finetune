import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from data_loader import dataloader
from function import imshow
from engine import Trainer
from engine_freeze import Trainer_freeze

cudnn.benchmark = True
plt.ion()   # interactive mode




if __name__ == '__main__':
    data = dataloader()
    trainer = Trainer_freeze(epochs=10, num_images=6)

    # Get a batch of training data
    inputs, classes = next(iter(data.dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[data.cls_name[x] for x in classes])

    model_ft = trainer.train_model()

    trainer.visualize_model()