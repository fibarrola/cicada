import clip
import torch
import pydiffvg
import torch
import torchvision.transforms as transforms
import time
import pickle
import datetime
import numpy as np
from torchvision import transforms
from src import versions, utils
from src.drawing import get_drawing_paths
from src.render_design import (
    add_shape_groups,
    load_vars,
    render_save_img,
    build_random_curves,
)
from src.style import VGG, image_loader, VGG2


# CLIP
versions.getinfo()
device = torch.device('cuda:0')
model, preprocess = clip.load('ViT-B/32', device, jit=False)
with open('data/nouns.txt', 'r') as f:
    nouns = f.readline()
    f.close()
nouns = nouns.split(" ")
noun_prompts = ["a drawing of a " + x for x in nouns]

clip_vectors = []
for i in range(4):
    path = 'data/dog_' + str(i) + '.png'
    img = image_loader(path).squeeze(0)
    img = img.permute(1, 2, 0)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=device
    ) * (1 - img[:, :, 3:4])
    im_batch = img.permute(2, 1, 0).unsqueeze(0)
    clip_vectors.append(model.encode_image(im_batch))

for k in range(3):
    print(torch.norm(clip_vectors[0] - clip_vectors[k + 1]).item())
