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

cudnn.benchmark = True
plt.ion()   # interactive mode


class Trainer():
    def __init__(self, epochs=25, num_images=6):
        self.save_path = './ckpt/last_ckpt.pt'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #
        self.epochs = epochs
        self.num_images = num_images
        #
        self.loader = dataloader()
        #
        self.model = self.get_model()
        #
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def get_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def get_optimizer(self):
        optimizer_ft = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optimizer_ft

    def get_scheduler(self):
        exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        return exp_lr_scheduler

    def get_model(self):
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        # labels of hymenoptera_data is 2 -> thus, out_features of pretrained model have to changed.
        model_ft = model_ft.to(self.device)
        return model_ft

    def train_model(self):
        since = time.time()

        # Create a temporary directory to save training checkpoints
        best_model_params_path = self.save_path

        torch.save(self.model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(self.epochs):
            print(f'Epoch {epoch}/{self.epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.loader.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.loader.len[phase]
                epoch_acc = running_corrects.double() / self.loader.len[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(torch.load(best_model_params_path))
        return self.model

    def visualize_model(self):
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.loader.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(self.num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {self.loader.cls_name[preds[j]]}')
                    imshow(inputs.cpu().data[j])

                    if images_so_far == self.num_images:
                        self.model.train(mode=was_training)
                        return
            self.model.train(mode=was_training)