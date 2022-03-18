from src.style import VGG, image_loader
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np

import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

style_model = VGG().to(device).eval()

# The overall idea is that two images in the same row column be closer to each other than two that share neither
# Moreover, columns one and two should be closer than columns two and three

style_features = [[[] for j in range(3)] for i in range(3)]

for i in range(3):
    for j in range(3):
        path = 'data/test_hat_'+str(i+1)+str(j+1)+'.png'
        img = image_loader(path).squeeze(0)
        img = img.permute(1,2,0)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = device) * (1 - img[:, :, 3:4])
        style_features[i][j] = style_model(img.permute(2,0,1))

for l in range(len(style_features[0][0])):
    (i1, j1) = (0,0)
    (i2, j2) = (0,1)
    shape_change = round(torch.norm(style_features[i1][j1][l] - style_features[i2][j2][l]).item() )
    (i1, j1) = (0,0)
    (i2, j2) = (1,0)
    color_change = round(torch.norm(style_features[i1][j1][l] - style_features[i2][j2][l]).item() )
    (i1, j1) = (0,0)
    (i2, j2) = (1,1)
    multi_change = round(torch.norm(style_features[i1][j1][l] - style_features[i2][j2][l]).item() )
    print(shape_change, color_change, multi_change, multi_change > color_change and multi_change > shape_change)

for l in range(len(style_features[0][0])):
    (i1, j1) = (0,0)
    (i2, j2) = (0,2)
    shape_change = round(torch.norm(style_features[i1][j1][l] - style_features[i2][j2][l]).item() )
    (i1, j1) = (0,0)
    (i2, j2) = (2,0)
    color_change = round(torch.norm(style_features[i1][j1][l] - style_features[i2][j2][l]).item() )
    (i1, j1) = (0,0)
    (i2, j2) = (2,2)
    multi_change = round(torch.norm(style_features[i1][j1][l] - style_features[i2][j2][l]).item() )
    print(shape_change, color_change, multi_change, multi_change > color_change and multi_change > shape_change)
