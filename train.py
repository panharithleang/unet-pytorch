import glob
import numpy as np
from torch import nn
from torch import optim
import torch
from torch.utils.data import DataLoader
from data_generator import DataGenerator
from model import UNet
import argparse
from tqdm import tqdm

LEARN_RATE = 0
BATCH_SIZE = 1
EPOCHS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device type: {device.type}')

parser = argparse.ArgumentParser()
parser.add_argument('--training_dir', type=str, help="path to train data ")
parser.add_argument('--validation_dir', type=str,
                    help="path to validation data")
parser.add_argument('--epoch', type=int, help="number of epoch")
parser.add_argument('--output_dir', type=str, help="path to output train")

args = parser.parse_args()
training_dir = args.training_dir
validation_dir = args.validation_dir
epoch = args.epoch
output_dir = args.output_dir


training_size = len(list(glob.glob(f'{training_dir}/*')))//2
training_data = DataGenerator(training_dir, training_size)

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
train_features, label_feature = next(iter(train_dataloader))

validation_size = len(list(glob.glob(f'{validation_dir}/*')))//2
validation_data = DataGenerator(validation_dir, validation_size)

validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=True)
train_features, label_feature = next(iter(validation_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {label_feature.size()}")
uNet = UNet(2)
uNet.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(uNet.parameters(), lr=0.001, momentum=0.99)


def train_loop(dataloader, model):
    size = dataloader.__len__()
    total_loss = 0
    with tqdm(total=size) as pbar:
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.update(1)

    return total_loss / size


def validation_loop(dataloader, model):
    size = dataloader.__len__()
    total_loss = 0
    with tqdm(total=size) as pbar:
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            pbar.update(1)
    return total_loss/size


for i in range(epoch):
    print(f'Epoch: {i}')
    epoch_train_loss = train_loop(train_dataloader, uNet)
    torch.save(uNet.state_dict(), f'{output_dir}/model_epoch_{i}.pth')
    epoch_validation_loss = validation_loop(validation_dataloader, uNet)
    print(
        f'train_loss: {epoch_train_loss}  validation_loss: {epoch_validation_loss}')
