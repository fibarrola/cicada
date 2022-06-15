from src import utils
from src.loss import CLIPConvLoss2
from src.processing import get_augment_trans
from src.render_design import (
    UserSketch,
    add_shape_groups,
    treebranch_initialization,
)
from src.svg_extraction import get_drawing_paths
import clip
from src.utils import get_nouns, shapes2paths
import torch
import pydiffvg
import copy

pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')


class DrawingModel:
    def __init__(self, args, device):
        self.device = device
        self.model, preprocess = clip.load('ViT-B/32', self.device, jit=False)
        self.clipConvLoss = CLIPConvLoss2(self.device)
        self.canvas_w = args.canvas_w
        self.canvas_h = args.canvas_h
        self.augment_trans = get_augment_trans(args.canvas_w, args.normalize_clip)
        self.shapes = []
        self.shape_groups = []
        self.path_list = []
        self.num_rnd_paths = 0

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

    def load_svg_shapes(self, args):
        '''This will discard all existing shapes'''
        self.path_list = get_drawing_paths(args.svg_path)
        user_sketch = UserSketch(args.canvas_w, args.canvas_h)
        user_sketch.build_shapes(self.path_list)
        self.shapes = user_sketch.shapes
        self.shape_groups = user_sketch.shape_groups
        self.num_sketch_paths = len(user_sketch.shapes)
        self.user_sketch = user_sketch

    def load_listed_shapes(self, args, shapes, shape_groups, tie=True):
        '''This will NOT discard existing shapes
        tie is a boolean indicating whether we penalize w.r.t. the added shapes'''
        self.path_list += shapes2paths(shapes, shape_groups, tie, args=args)
        self.shapes += shapes
        self.shape_groups = add_shape_groups(self.shape_groups, shape_groups)
        self.user_sketch = UserSketch(args.canvas_w, args.canvas_h)
        self.user_sketch.build_shapes(self.path_list)
        self.num_sketch_paths = len(self.user_sketch.shapes)

    def add_random_shapes(self, num_rnd_paths, args):
        '''This will NOT discard existing shapes'''
        self.num_rnd_paths += num_rnd_paths
        shapes_rnd, shape_groups_rnd = treebranch_initialization(
            self.path_list,
            num_rnd_paths,
            args.canvas_w,
            args.canvas_h,
            args.drawing_area,
        )
        self.path_list += shapes2paths(
            shapes_rnd, shape_groups_rnd, tie=False, args=args
        )
        self.shapes = self.shapes + shapes_rnd
        self.shape_groups = add_shape_groups(self.shape_groups, shape_groups_rnd)

    def remove_traces(self, idx_list, args):
        '''Remove the traces indexed in idx_list'''
        self.shapes = [
            self.shapes[k] for k in range(len(self.shapes)) if k not in idx_list
        ]
        self.shape_groups = add_shape_groups(
            [
                self.shape_groups[k]
                for k in range(len(self.shape_groups))
                if k not in idx_list
            ],
            [],
        )
        self.path_list = [
            self.path_list[k] for k in range(len(self.path_list)) if k not in idx_list
        ]
        self.initialize_variables(args)

    def initialize_variables(self, args):
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
        self.user_sketch.init_vars()
        self.points_vars0 = copy.deepcopy(self.points_vars)
        self.stroke_width_vars0 = copy.deepcopy(self.stroke_width_vars)
        self.color_vars0 = copy.deepcopy(self.color_vars)
        for k in range(len(self.color_vars0)):
            self.points_vars0[k].requires_grad = False
            self.stroke_width_vars0[k].requires_grad = False
            self.color_vars0[k].requires_grad = False
        self.img0 = copy.deepcopy(self.user_sketch.img)

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

        for k in range(len(self.points_vars)):
            if self.path_list[k].is_tied:
                points_loss += torch.norm(self.points_vars[k] - self.points_vars0[k])
                colors_loss += torch.norm(self.color_vars[k] - self.color_vars0[k])
                widths_loss += torch.norm(
                    self.stroke_width_vars[k] - self.stroke_width_vars0[k]
                )

        loss += args.w_points * points_loss
        loss += args.w_colors * colors_loss
        loss += args.w_widths * widths_loss
        loss += args.w_img * img_loss

        geo_loss = self.clipConvLoss(img * self.mask + 1 - self.mask, self.img0)

        for l_name in geo_loss:
            loss += args.w_geo * geo_loss[l_name]
        # loss += args.w_geo * geo_loss['clip_conv_loss_layer3']

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

    def prune(self, prune_ratio, args):
        with torch.no_grad():

            # Get points of tied traces
            tied_points = []
            for p, path in enumerate(self.path_list):
                if path.is_tied:
                    tied_points += [
                        x.unsqueeze(0)
                        for i, x in enumerate(self.shapes[p].points)
                        if i % 3 == 0
                    ]  # only points the path goes through

            # Compute distances
            dists = []
            if tied_points:
                tied_points = torch.cat(tied_points, 0)
                for p, path in enumerate(self.path_list):
                    if path.is_tied:
                        dists.append(-1000)
                    else:
                        points = [
                            x.unsqueeze(0)
                            for i, x in enumerate(self.shapes[p].points)
                            if i % 3 == 0
                        ]  # only points the path goes through
                        min_dists = []
                        for point in points:
                            d = torch.norm(point - tied_points, dim=1)
                            d = min(d)
                            min_dists.append(d.item())

                        dists.append(min(min_dists))

            # Compute losses
            losses = []
            for p, path in enumerate(self.path_list):
                if path.is_tied:
                    losses.append(1000)
                else:
                    # Compute the loss if we take out the k-th path
                    shapes = self.shapes[:p] + self.shapes[p + 1 :]
                    shape_groups = add_shape_groups(
                        self.shape_groups[:p], self.shape_groups[p + 1 :]
                    )
                    img = self.build_img(shapes, shape_groups, 5)
                    img_augs = []
                    for n in range(args.num_augs):
                        img_augs.append(self.augment_trans(img))
                    im_batch = torch.cat(img_augs)
                    img_features = self.model.encode_image(im_batch)
                    loss = 0
                    for n in range(args.num_augs):
                        loss -= torch.cosine_similarity(
                            self.text_features, img_features[n : n + 1], dim=1
                        )
                    losses.append(loss.cpu().item())

            # Compute scores
            scores = [-0.01 * dists[k] ** (0.5) + losses[k] for k in range(len(losses))]

            # Actual pruning
            inds = utils.k_max_elements(scores, int((1 - prune_ratio) * args.num_paths))

            # Define the lists like this because using "for p in inds"
            # may (and often will) change the order of the traces
            self.shapes = [
                self.shapes[p] for p in range(len(self.path_list)) if p in inds
            ]
            self.shape_groups = add_shape_groups(
                [self.shape_groups[p] for p in range(len(self.path_list)) if p in inds],
                [],
            )
            self.path_list = [
                self.path_list[p] for p in range(len(self.path_list)) if p in inds
            ]

        self.initialize_variables(args)
