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
from src.render_design import add_shape_groups, load_vars, render_save_img, build_random_curves
from src.style import VGG


   
# Parameters 
params = lambda: None
# params.svg_path: ''
params.clip_prompt = 'A drawing of a plant.'
params.neg_prompt = 'A badly drawn sketch.'
params.neg_prompt_2 = 'Many ugly, messy drawings.'
params.use_neg_prompts = False
params.normalize_clip = True
params.num_paths = 64
params.canvas_h = 224
params.canvas_w = 224
params.num_iter = 1000
params.max_width = 40
params.num_trials = 1

versions.getinfo()
device = torch.device('cuda:0')

style_model = VGG().to(device).eval()


# Pre-processing

model, preprocess = clip.load('ViT-B/32', device, jit=False)
with open('data/nouns.txt', 'r') as f:
    nouns = f.readline()
    f.close()
nouns = nouns.split(" ")
noun_prompts = ["a drawing of a " + x for x in nouns]


for noun in ['mask', 'key', 'bee', 'guitar', 'plane', 'shoe', 'surfboard', 'tank']:
    params.clip_prompt = f'A drawing of a {noun}.'

    for trial in range(params.num_trials):
        time_str = noun + '_{number:02d}'.format(number=trial)

        text_input = clip.tokenize(params.clip_prompt).to(device)
        text_input_neg1 = clip.tokenize(params.neg_prompt).to(device)
        text_input_neg2 = clip.tokenize(params.neg_prompt_2).to(device)

        with torch.no_grad():
            nouns_features = model.encode_text(torch.cat([clip.tokenize(noun_prompts).to(device)]))
            text_features = model.encode_text(text_input)
            text_features_neg1 = model.encode_text(text_input_neg1)
            text_features_neg2 = model.encode_text(text_input_neg2)

        pydiffvg.set_print_timing(False)
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        pydiffvg.set_device(device)

        # Image Augmentation Transformation
        augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(params.canvas_w, scale=(0.7,0.9)),
        ])

        if params.normalize_clip:
            augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(params.canvas_w, scale=(0.7,0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        shapes, shape_groups = build_random_curves(
            params.num_paths,
            params.canvas_w,
            params.canvas_h,
            0,
            1,
            0,
            1,
            )
    
        points_vars0, stroke_width_vars0, color_vars0, img0 = load_vars()

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

        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            params.canvas_w, params.canvas_h, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply


        # Optimizers
        points_optim = torch.optim.Adam(points_vars, lr=0.5)
        width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
        color_optim = torch.optim.Adam(color_vars, lr=0.01)

        # Run the main optimization loop
        for t in range(params.num_iter):

            points_optim.zero_grad()
            width_optim.zero_grad()
            color_optim.zero_grad()
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                params.canvas_w, params.canvas_h, shapes, shape_groups)
            img = render(params.canvas_w, params.canvas_h, 2, 2, t, None, *scene_args)
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])

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
                if params.use_neg_prompts:
                    loss += torch.cosine_similarity(text_features_neg1, image_features[n:n+1], dim=1) * 0.3
                    loss += torch.cosine_similarity(text_features_neg2, image_features[n:n+1], dim=1) * 0.3


            # Backpropagate the gradients.
            loss.backward()

            # Take a gradient descent step.
            points_optim.step()
            width_optim.step()
            color_optim.step()
            for path in shapes:
                path.stroke_width.data.clamp_(1.0, params.max_width)
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)
            
            if t % 50 == 0:
                # utils_def.show_img(img.detach().cpu().numpy()[0])
                # show_img(torch.cat([img.detach(), img_aug.detach()], axis=3).cpu().numpy()[0])
                print('render loss:', loss.item())
                print('iteration:', t)
                with torch.no_grad():
                    # pydiffvg.imwrite(img.cpu().permute(0, 2, 3, 1).squeeze(0), 'results/'+time_str+'.png', gamma=1)

                    im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                    noun_norm = nouns_features / nouns_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(5)
                    print("\nTop predictions:\n")
                    for value, index in zip(values, indices):
                        print(f"{nouns[index]:>16s}: {100 * value.item():.2f}%")

                    # top_prediction_list[trial] = [nouns[indices[0]], values[0].item()]

        pydiffvg.imwrite(img.cpu().permute(0, 2, 3, 1).squeeze(0), 'results/'+time_str+'.png', gamma=1)
        
        # style_features = style_model(img)

        # with open('results/'+time_str+'_style.pkl', 'wb') as f:
        #     pickle.dump([x.detach().cpu() for x in style_features], f)

        # utils.save_data(time_str, params)

    # X = []
    # rate = 0
    # for pred in top_prediction_list:
    #     if pred[0] == 'hat':
    #         rate+=1
    #     X.append(pred[1])
    # print(top_prediction_list)

    # print(np.mean(X), np.std(X))