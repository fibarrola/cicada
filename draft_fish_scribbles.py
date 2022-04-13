import clip
import torch
import pydiffvg
import datetime
from src import versions, utils
from src.svg_extraction import get_drawing_paths
from src.render_design import (
    add_shape_groups,
    load_vars,
    render_save_img,
    treebranch_initialization,
)
from src.processing import get_augment_trans
from config import args
from src.loss import CLIPConvLoss2
from src.style import image_loader

versions.getinfo(showme=False)
device = torch.device('cuda:0')
clipConvLoss = CLIPConvLoss2(device)
colors = ['red', 'usydred', 'secondary']

for i in range(3):
    path = f'data/cartoons/partialcartoon_{i}.png'
    img = image_loader(path).squeeze(0)
    img = img.permute(1, 2, 0)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=device
    ) * (1 - img[:, :, 3:4])
    img0 = img.permute(2, 1, 0).unsqueeze(0).to(device).clone()

    for img_name in ['scribbled', 'scribbled2', 'full']:
        path = f'data/cartoons/{img_name}cartoon_{i}.png'
        img = image_loader(path).squeeze(0)
        img = img.permute(1, 2, 0)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=device
        ) * (1 - img[:, :, 3:4])
        loss = clipConvLoss(img.permute(2, 1, 0).unsqueeze(0).to(device), img0)
        print(loss)
        losses = [
            '\\'
            + 'textcolor{'
            + colors[c]
            + '}{'
            + '{loss:1.5f}'.format(loss=loss[l].item())
            + '}, '
            for c, l in enumerate(loss)
        ]
        string = ''
        for l in losses:
            string += l
        print(img_name[:4], string)

    print(' ')



print(' ')
print(' ')
print(' ')

path = f'data/positions/position_0.png'
img = image_loader(path).squeeze(0)
img = img.permute(1, 2, 0)
img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
    img.shape[0], img.shape[1], 3, device=device
) * (1 - img[:, :, 3:4])
img0 = img.permute(2, 1, 0).unsqueeze(0).to(device).clone()

for position in range(4):
    path = f'data/positions/position_{position}.png'
    img = image_loader(path).squeeze(0)
    img = img.permute(1, 2, 0)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=device
    ) * (1 - img[:, :, 3:4])
    loss = clipConvLoss(img.permute(2, 1, 0).unsqueeze(0).to(device), img0)
    losses = [
        '\\'
        + 'textcolor{'
        + colors[c]
        + '}{'
        + '{loss:1.4f}'.format(loss=loss[l].item())
        + '}, '
        for c, l in enumerate(loss)
    ]
    string = ''
    for l in losses:
        string += l
    print(string)

print(' ')
print(' ')
print(' ')

path = f'data/positions/duck_bot_left.png'
img = image_loader(path).squeeze(0)
img = img.permute(1, 2, 0)
img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
    img.shape[0], img.shape[1], 3, device=device
) * (1 - img[:, :, 3:4])
img0 = img.permute(2, 1, 0).unsqueeze(0).to(device).clone()

for im_name in ['duck_center_center', 'duck_top_right']:
    path = f'data/positions/{im_name}.png'
    img = image_loader(path).squeeze(0)
    img = img.permute(1, 2, 0)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=device
    ) * (1 - img[:, :, 3:4])
    loss = clipConvLoss(img.permute(2, 1, 0).unsqueeze(0).to(device), img0)
    losses = [
        '\\'
        + 'textcolor{'
        + colors[c]
        + '}{'
        + '{loss:1.4f}'.format(loss=loss[l].item())
        + '}, '
        for c, l in enumerate(loss)
    ]
    string = ''
    for l in losses:
        string += l
    print(string)
    print(torch.norm(img.permute(2, 1, 0).unsqueeze(0).to(device)-img0))