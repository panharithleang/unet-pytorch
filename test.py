from torch.utils.data.dataloader import DataLoader
from data_generator import DataGenerator
import torch
import numpy as np
import cv2

training_data = DataGenerator(
    '/Users/leangpanharith/Documents/school_stuffs/unet/data/training', 800)

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)

X, y = next(iter(train_dataloader))

x = X[0].permute(1, 2, 0)

img = np.array(x, dtype=np.uint8)
mask = np.array(y[0][0], dtype=np.uint8)
# print(mask.shape)
# print(np.min(mask))
# print(np.max(mask))
cv2.imshow('preview', img)
cv2.imshow('mask', mask)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
# print(X[0].size())
# print(y[0].size())
