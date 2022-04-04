from src import utils
from src.loss import CLIPConvLoss2
from src.processing import get_augment_trans
from src.render_design import (
    add_shape_groups,
    load_vars,
    render_save_img,
    treebranch_initialization,
)
from src.svg_extraction import get_drawing_paths
import clip
from src.utils import get_nouns
import torch
import pydiffvg

pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')


class DrawingModel:
    def __init__(self, args, device):
        self.device = device
        self.path_list = get_drawing_paths(args.svg_path)
        self.model, preprocess = clip.load('ViT-B/32', self.device, jit=False)
        self.clipConvLoss = CLIPConvLoss2(self.device)

    def process_text(self, args):
        self.nouns, noun_prompts = get_nouns()
        text_input = clip.tokenize(args.clip_prompt).to(self.device)
        text_input_neg1 = clip.tokenize(args.neg_prompt).to(self.device)
        text_input_neg2 = clip.tokenize(args.neg_prompt_2).to(self.device)
        with torch.no_grad():
            self.nouns_features = self.model.encode_text(
                torch.cat([clip.tokenize(noun_prompts).to(self.device)])
            )
            self.text_features = self.model.encode_text(text_input)
            self.text_features_neg1 = self.model.encode_text(text_input_neg1)
            self.text_features_neg2 = self.model.encode_text(text_input_neg2)

    def initialize_variables(self, args):
        self.canvas_w = args.canvas_w
        self.canvas_h = args.canvas_h
        shapes_sketch, shape_groups_sketch = render_save_img(
            self.path_list, args.canvas_w, args.canvas_h
        )
        shapes_rnd, shape_groups_rnd = treebranch_initialization(
            self.path_list,
            args.num_paths,
            args.canvas_w,
            args.canvas_h,
            args.drawing_area,
        )
        self.shapes = shapes_sketch + shapes_rnd
        self.shape_groups = add_shape_groups(shape_groups_sketch, shape_groups_rnd)
        self.num_sketch_paths = len(shapes_sketch)
        self.augment_trans = get_augment_trans(args.canvas_w, args.normalize_clip)

        self.points_vars = []
        self.stroke_width_vars = []
        self.color_vars = []
        for path in self.shapes:
            path.points.requires_grad = True
            self.points_vars.append(path.points)
            path.stroke_width.requires_grad = True
            self.stroke_width_vars.append(path.stroke_width)
        for group in self.shape_groups:
            group.stroke_color.requires_grad = True
            self.color_vars.append(group.stroke_color)

        self.render = pydiffvg.RenderFunction.apply
        self.mask = utils.area_mask(args.canvas_w, args.canvas_h, args.drawing_area).to(
            self.device
        )
        (
            self.points_vars0,
            self.stroke_width_vars0,
            self.color_vars0,
            self.img0,
        ) = load_vars()

    def initialize_optimizer(self):
        self.points_optim = torch.optim.Adam(self.points_vars, lr=0.5)
        self.width_optim = torch.optim.Adam(self.stroke_width_vars, lr=0.1)
        self.color_optim = torch.optim.Adam(self.color_vars, lr=0.01)

    def build_img(self, shapes, shape_groups, t):
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_w, self.canvas_h, shapes, shape_groups
        )
        img = self.render(self.canvas_w, self.canvas_h, 2, 2, t, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img

    def run_epoch(self, t, args):
        self.points_optim.zero_grad()
        self.width_optim.zero_grad()
        self.color_optim.zero_grad()

        img = self.build_img(self.shapes, self.shape_groups, t)

        img_loss = (
            torch.norm((img - self.img0) * self.mask)
            if args.w_img > 0
            else torch.tensor(0, device=self.device)
        )

        self.img = img.cpu().permute(0, 2, 3, 1).squeeze(0)

        loss = 0

        img_augs = []
        for n in range(args.num_augs):
            img_augs.append(self.augment_trans(img))
        im_batch = torch.cat(img_augs)
        img_features = self.model.encode_image(im_batch)
        for n in range(args.num_augs):
            loss -= torch.cosine_similarity(
                self.text_features, img_features[n : n + 1], dim=1
            )
            if args.use_neg_prompts:
                loss += (
                    torch.cosine_similarity(
                        self.text_features_neg1, img_features[n : n + 1], dim=1
                    )
                    * 0.3
                )
                loss += (
                    torch.cosine_similarity(
                        self.text_features_neg2, img_features[n : n + 1], dim=1
                    )
                    * 0.3
                )
        self.img_features = img_features

        points_loss = 0
        widths_loss = 0
        colors_loss = 0

        for k, points0 in enumerate(self.points_vars0):
            points_loss += torch.norm(self.points_vars[k] - points0)
            colors_loss += torch.norm(self.color_vars[k] - self.color_vars0[k])
            widths_loss += torch.norm(
                self.stroke_width_vars[k] - self.stroke_width_vars0[k]
            )

        loss += args.w_points * points_loss
        loss += args.w_colors * colors_loss
        loss += args.w_widths * widths_loss
        loss += args.w_img * img_loss

        geo_loss = self.clipConvLoss(img, self.img0)

        for l_name in geo_loss:
            loss += args.w_geo * geo_loss[l_name]

        if args.w_geo_p > 0:
            img_p = self.build_img(
                self.shapes[: self.num_sketch_paths],
                self.shape_groups[: self.num_sketch_paths],
                t,
            )
            geo_loss = self.clipConvLoss(img_p, self.img0)
            for l_name in geo_loss:
                loss += args.w_geo_p * geo_loss[l_name]

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        self.points_optim.step()
        self.width_optim.step()
        self.color_optim.step()
        for path in self.shapes:
            path.stroke_width.data.clamp_(1.0, args.max_width)
        for group in self.shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

        self.losses = {
            'global': loss,
            'points': points_loss,
            'widhts': widths_loss,
            'colors': colors_loss,
            'image': img_loss,
            'geometric': geo_loss,
        }
