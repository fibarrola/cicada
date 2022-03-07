import os
from src import utils_def, versions
import clip
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR100

from src.drawing import get_drawing_paths
from src.render_design import render_save_img


# a, b = versions.get_versions()
# print(a, b)
# import torch

import sys
print('A', sys.version)
print('B', torch.__version__)
print('C', torch.cuda.is_available())
print('D', torch.backends.cudnn.enabled)
device = torch.device('cuda')
print('E', torch.cuda.get_device_properties(device))
print('F', torch.tensor([1.0, 2.0]).cuda())

# Load the model
device = torch.device('cuda:0')
model, preprocess = clip.load('ViT-B/32', device, jit=False)
with open('data/nouns.txt', 'r') as f:
    nouns = f.readline()
    f.close()
nouns = nouns.split(" ")
noun_prompts = ["a drawing of a " + x for x in nouns]

# Calculate features
with torch.no_grad():
    nouns_features = model.encode_text(torch.cat([clip.tokenize(noun_prompts).to(device)]))
print(nouns_features.shape, nouns_features.dtype)



#@title Initializing {vertical-output: true}


device = torch.device('cuda:0')

# LOAD THE DRAWING
path_to_svg_file = 'data/drawing2.svg'
path_list = get_drawing_paths(path_to_svg_file)

# %cd /content/diffvg/apps/

prompt = "A red chair."
neg_prompt = "A badly drawn sketch."
neg_prompt_2 = "Many ugly, messy drawings."
text_input = clip.tokenize(prompt).to(device)
text_input_neg1 = clip.tokenize(neg_prompt).to(device)
text_input_neg2 = clip.tokenize(neg_prompt_2).to(device)
use_negative = False # Use negative prompts?
use_normalized_clip = True 

# Calculate features
with torch.no_grad():
    text_features = model.encode_text(text_input)
    text_features_neg1 = model.encode_text(text_input_neg1)
    text_features_neg2 = model.encode_text(text_input_neg2)

import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import pickle

pydiffvg.set_print_timing(False)

gamma = 1.0

# ARGUMENTS. Feel free to play around with these, especially num_paths.
args = lambda: None
args.num_paths = len(path_list) + 4
args.num_iter = 1500
args.max_width = 65
args.lambda_points = 0.01
args.lambda_colors = 0.1
args.lambda_widths = 0.01
args.lambda_img = 0.01
args.t1 = 1500
args.lambda_full_img = 0.001

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
device = torch.device('cuda')
pydiffvg.set_device(device)

canvas_width, canvas_height = 224, 224
num_paths = args.num_paths
max_width = args.max_width

# Image Augmentation Transformation
augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
])

if use_normalized_clip:
    augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

shapes, shape_groups = render_save_img(path_list, canvas_width, canvas_height)

# Add random curves
for i in range(num_paths-len(path_list)):
    num_segments = random.randint(1, 3)
    num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
    points = []
    p0 = (random.random(), random.random())
    points.append(p0)
    for j in range(num_segments):
        radius = 0.1
        p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
        p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
        p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
        points.append(p1)
        points.append(p2)
        points.append(p3)
        p0 = p3
    points = torch.tensor(points)
    points[:, 0] *= canvas_width
    points[:, 1] *= canvas_height
    path = pydiffvg.Path(num_control_points = num_control_points, points = points, stroke_width = torch.tensor(1.0), is_closed = False)
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
    shape_groups.append(path_group)

points_vars = []
stroke_width_vars = []
color_vars = []
for path in shapes:
    path.points.requires_grad = True
    points_vars.append(path.points)
    path.stroke_width.requires_grad = True
    stroke_width_vars.append(path.stroke_width)
for group in shape_groups:
    group.stroke_color.requires_grad = True
    color_vars.append(group.stroke_color)

# Just some diffvg setup
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
render = pydiffvg.RenderFunction.apply
img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
img = img[:, :, :3]
img = img.unsqueeze(0)
img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
# utils_def.show_img(img.detach().cpu().numpy()[0])

import datetime

with open('tmp/points_vars.pkl', 'rb') as f:
    points_vars0 = pickle.load(f)
with open('tmp/stroke_width_vars.pkl', 'rb') as f:
    stroke_width_vars0 = pickle.load(f)
with open('tmp/color_vars.pkl', 'rb') as f:
    color_vars0 = pickle.load(f)
with open('tmp/img0.pkl', 'rb') as f:
    img0 = pickle.load(f)

# Optimizers
points_optim = torch.optim.Adam(points_vars, lr=1.0)
width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
color_optim = torch.optim.Adam(color_vars, lr=0.01)

# Run the main optimization loop
for t in range(args.num_iter):

    # Anneal learning rate (makes videos look cleaner)
    if t == int(args.num_iter * 0.5):
        for g in points_optim.param_groups:
            g['lr'] = 0.4
    if t == int(args.num_iter * 0.75):
        for g in points_optim.param_groups:
            g['lr'] = 0.1
    
    points_optim.zero_grad()
    width_optim.zero_grad()
    color_optim.zero_grad()
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])

    l_img = torch.norm(img[canvas_height//2:,:,:]-img0[canvas_height//2:,:,:])
    if t > args.t1:
        l_full_img = args.lambda_full_img*torch.norm(torch.ones_like(img)-img)
    else:
        l_full_img = 0

    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2) # NHWC -> NCHW

    loss = 0
    

    NUM_AUGS = 4
    img_augs = []
    for n in range(NUM_AUGS):
        img_augs.append(augment_trans(img))
    im_batch = torch.cat(img_augs)
    image_features = model.encode_image(im_batch)
    for n in range(NUM_AUGS):
        loss -= torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)
        if use_negative:
            loss += torch.cosine_similarity(text_features_neg1, image_features[n:n+1], dim=1) * 0.3
            loss += torch.cosine_similarity(text_features_neg2, image_features[n:n+1], dim=1) * 0.3

    l_points = 0
    l_widths = 0
    l_colors = 0
    style_images = [0 for k in range(4)]
    l_style = torch.tensor([0 for im in style_images])

    # # don't do this every time
    # for k, im in enumerate(style_images):
    #     gen_features=style_model(img)
    #     style_features=style_model(im)
    #     for gen,style in zip(gen_features, style_features):
    #         l_style[k] += calc_style_loss(gen, style)
        
    for k, points0 in enumerate(points_vars0):
        l_points += torch.norm(points_vars[k]-points0)
        l_colors += torch.norm(color_vars[k]-color_vars0[k])
        l_widths += torch.norm(stroke_width_vars[k]-stroke_width_vars0[k])
    
    loss += args.lambda_points*l_points
    loss += args.lambda_colors*l_colors
    loss += args.lambda_widths*l_widths
    loss += args.lambda_img*l_img
    loss += l_full_img

    

    # Backpropagate the gradients.
    loss.backward()

    # Take a gradient descent step.
    points_optim.step()
    width_optim.step()
    color_optim.step()
    for path in shapes:
        path.stroke_width.data.clamp_(1.0, max_width)
    for group in shape_groups:
        group.stroke_color.data.clamp_(0.0, 1.0)
    
    if t % 50 == 0:
        # utils_def.show_img(img.detach().cpu().numpy()[0])
        # show_img(torch.cat([img.detach(), img_aug.detach()], axis=3).cpu().numpy()[0])
        print('render loss:', loss.item())
        print('l_points: ', l_points.item())
        print('l_colors: ', l_colors.item())
        print('l_widths: ', l_widths.item())
        print('l_img: ', l_img.item())
        for l in l_style:
            print('l_style: ', l.item())
        print('iteration:', t)
        with torch.no_grad():
            im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            noun_norm = nouns_features / nouns_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)
            print("\nTop predictions:\n")
            for value, index in zip(values, indices):
                print(f"{nouns[index]:>16s}: {100 * value.item():.2f}%")

time_str = (datetime.datetime.today() + datetime.timedelta(hours = 11)).strftime("%Y_%m_%d_%H_%M")
img = img.permute(0, 2, 3, 1)
img = img.squeeze(0)
pydiffvg.imwrite(img.cpu(), 'results/'+time_str+'.png', gamma=1)
with open('results/'+time_str+'.txt', 'w') as f:
    f.write('I0: ' +path_to_svg_file +'\n')
    f.write('prompt: ' +str(prompt) +'\n')
    f.write('num paths: ' +str(args.num_paths) +'\n')
    f.write('num_iter: ' +str(args.num_iter) +'\n')
    f.write('lambda_points: '+str(args.lambda_points)+'\n')
    f.write('lambda_colors: '+str(args.lambda_colors)+'\n')
    f.write('lambda_widths: '+str(args.lambda_widths)+'\n')
    f.write('lambda_img: '+str(args.lambda_img)+'\n')