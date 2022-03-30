import clip
import torch
import pydiffvg
import datetime
import numpy as np
from torchvision import transforms
from src import versions, utils
from src.svg_extraction import get_drawing_paths
from src.render_design import (
    add_shape_groups,
    load_vars,
    render_save_img,
    treebranch_initialization,
)
from config import args


versions.getinfo(showme=False)
device = torch.device('cuda:0')


# Pre-processing
model, preprocess = clip.load('ViT-B/32', device, jit=False)
with open('data/nouns.txt', 'r') as f:
    nouns = f.readline()
    f.close()
nouns = nouns.split(" ")
noun_prompts = ["a drawing of a " + x for x in nouns]

for trial in range(args.num_trials):
    time_str = (datetime.datetime.today() + datetime.timedelta(hours=11)).strftime(
        "%Y_%m_%d_%H_%M_%S"
    )

    path_list = get_drawing_paths(args.svg_path)
    text_input = clip.tokenize(args.clip_prompt).to(device)
    text_input_neg1 = clip.tokenize(args.neg_prompt).to(device)
    text_input_neg2 = clip.tokenize(args.neg_prompt_2).to(device)

    with torch.no_grad():
        nouns_features = model.encode_text(
            torch.cat([clip.tokenize(noun_prompts).to(device)])
        )
        text_features = model.encode_text(text_input)
        text_features_neg1 = model.encode_text(text_input_neg1)
        text_features_neg2 = model.encode_text(text_input_neg2)

    pydiffvg.set_print_timing(False)
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_device(device)

    # Image Augmentation Transformation
    augment_trans = transforms.Compose(
        [
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(args.canvas_w, scale=(0.7, 0.9)),
        ]
    )

    if args.normalize_clip:
        augment_trans = transforms.Compose(
            [
                transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
                transforms.RandomResizedCrop(args.canvas_w, scale=(0.7, 0.9)),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    shapes, shape_groups = render_save_img(path_list, args.canvas_w, args.canvas_h)
    shapes_rnd, shape_groups_rnd = treebranch_initialization(
        path_list, args.num_paths, args.canvas_w, args.canvas_h, args.drawing_area,
    )
    shapes += shapes_rnd
    shape_groups = add_shape_groups(shape_groups, shape_groups_rnd)

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

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        args.canvas_w, args.canvas_h, shapes, shape_groups
    )
    render = pydiffvg.RenderFunction.apply

    mask = utils.area_mask(args.canvas_w, args.canvas_h, args.drawing_area).to(device)

    # Optimizers
    points_optim = torch.optim.Adam(points_vars, lr=0.5)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    random_inds = np.random.randint(2, size=len(points_vars))
    random_inds = [x == 0 for x in random_inds]

    # Run the main optimization loop
    for t in range(args.num_iter):

        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            args.canvas_w, args.canvas_h, shapes, shape_groups
        )
        img = render(args.canvas_w, args.canvas_h, 2, 2, t, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()
        ) * (1 - img[:, :, 3:4])

        if args.w_img > 0:
            l_img = torch.norm((img - img0) * mask)
        else:
            l_img = torch.tensor(0, device=device)

        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        loss = 0

        NUM_AUGS = 4
        img_augs = []
        for n in range(NUM_AUGS):
            img_augs.append(augment_trans(img))
        im_batch = torch.cat(img_augs)
        img_features = model.encode_image(im_batch)
        for n in range(NUM_AUGS):
            loss -= torch.cosine_similarity(
                text_features, img_features[n : n + 1], dim=1
            )
            if args.use_neg_prompts:
                loss += (
                    torch.cosine_similarity(
                        text_features_neg1, img_features[n : n + 1], dim=1
                    )
                    * 0.3
                )
                loss += (
                    torch.cosine_similarity(
                        text_features_neg2, img_features[n : n + 1], dim=1
                    )
                    * 0.3
                )

        l_points = 0
        l_widths = 0
        l_colors = 0

        for k, points0 in enumerate(points_vars0):
            l_points += torch.norm(points_vars[k] - points0)
            l_colors += torch.norm(color_vars[k] - color_vars0[k])
            l_widths += torch.norm(stroke_width_vars[k] - stroke_width_vars0[k])

        loss += args.w_points * l_points
        loss += args.w_colors * l_colors
        loss += args.w_widths * l_widths
        loss += args.w_img * l_img

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        width_optim.step()
        color_optim.step()
        for path in shapes:
            path.stroke_width.data.clamp_(1.0, args.max_width)
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

        if t == 0:
            with torch.no_grad():
                pydiffvg.imwrite(
                    img.cpu().permute(0, 2, 3, 1).squeeze(0),
                    'results/' + time_str + '_0.png',
                    gamma=1,
                )

        if t % 50 == 0:
            print('render loss:', loss.item())
            print('l_points: ', l_points.item())
            print('l_colors: ', l_colors.item())
            print('l_widths: ', l_widths.item())
            print('l_img: ', l_img.item())
            print('iteration:', t)
            with torch.no_grad():
                pydiffvg.imwrite(
                    img.cpu().permute(0, 2, 3, 1).squeeze(0),
                    'results/' + time_str + '.png',
                    gamma=1,
                )
                im_norm = img_features / img_features.norm(dim=-1, keepdim=True)
                noun_norm = nouns_features / nouns_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
                values, indices = similarity[0].topk(5)
                print("\nTop predictions:\n")
                for value, index in zip(values, indices):
                    print(f"{nouns[index]:>16s}: {100 * value.item():.2f}%")

    pydiffvg.imwrite(
        img.cpu().permute(0, 2, 3, 1).squeeze(0),
        'results/' + time_str + '.png',
        gamma=1,
    )
    utils.save_data(time_str, args)
