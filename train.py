import glob
import cv2
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from data_generator import DataGenerator
from model import UNet
import argparse

LEARN_RATE = 0
BATCH_SIZE = 1
EPOCHS = 5

parser = argparse.ArgumentParser()
parser.add_argument('training_dir', type=str, help="path to data dir")

training_dir = "/Users/leangpanharith/Documents/school_stuffs/unet/data/training"

training_size = len(list(glob.glob(f'{training_dir}/*')))//2
training_data = DataGenerator(training_dir, training_size)

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
train_features, label_feature = next(iter(train_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {label_feature.size()}")
uNet = UNet(2)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(uNet.parameters(), lr=0.001, momentum=0.99)


def train_loop(dataloader, model):
    size = dataloader.__len__()
    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train_loop(train_dataloader, uNet)
