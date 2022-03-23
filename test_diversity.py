from src.style import VGG, image_loader
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np

import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

style_model = VGG().to(device).eval()

stdevs_0 = []
stdevs_1 = []

for layer in range(5):
    X0 = []

    for k in range(10):
        path = 'data/hat_{number:02d}.png'.format(number=k+1)

        img = image_loader(path).squeeze(0)

        img = img.permute(1,2,0)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = device) * (1 - img[:, :, 3:4])


        style_features = style_model(img.permute(2,0,1))

        X0.append(style_features[layer].view((-1,1)).detach().cpu())

    X0 = np.transpose(np.concatenate(X0,1))

    X1 = []

    for k in range(10):
        path = 'data/{number:02d}.png'.format(number=k+1)

        img = image_loader(path).squeeze(0)

        img = img.permute(1,2,0)


        if img.size(2) == 4:
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = device) * (1 - img[:, :, 3:4])

        style_features = style_model(img.permute(2,0,1))

        X1.append(style_features[layer].view((-1,1)).detach().cpu())

    X1 = np.transpose(np.concatenate(X1,1))

    print(np.mean(np.std(X0, axis = 0)),np.mean(np.std(X1, axis = 0)))
    stdevs_0.append(np.mean(np.std(X0, axis = 0)))
    stdevs_1.append(np.mean(np.std(X1, axis = 0)))
# X = np.concatenate([X0, X1], 0)

# print(X.shape)

# pca = PCA(n_components=2)
# print(pca.fit(X))

# Xp = pca.transform(X)

# print(Xp)

# plt.scatter(Xp[:10,0], Xp[:10,1])
# plt.scatter(Xp[10:20,0], Xp[10:20,1])

plt.plot(stdevs_0, 'o')
plt.plot(stdevs_1, 'o')
plt.box(on=False)
plt.legend(['User-drawn','Agent-drawn'])
plt.xlabel('layer')
plt.xticks(ticks=range(5), labels=['0','5','10','19','28']) 
plt.ylabel('mean stdev')


# # plt.imshow(img.cpu())
plt.show()