import torch.backends.cudnn as cudnn
import torchvision
import matplotlib.pyplot as plt
from data_loader import dataloader
from function import imshow
from engine.engine import Trainer
from engine.engine_freeze import Trainer_freeze

cudnn.benchmark = True
plt.ion()   # interactive mode




if __name__ == '__main__':
    data = dataloader()
    trainer = Trainer(epochs=25, num_images=6)

    # Get a batch of training data
    inputs, classes = next(iter(data.dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[data.cls_name[x] for x in classes])

    model_ft = trainer.train_model()

    trainer.visualize_model()

    trainer.model_predict()

    trainer.print_loss()