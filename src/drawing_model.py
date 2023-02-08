from src import utils
from src.loss import CLIPConvLoss2
from src.render_design import treebranch_initialization
from src.drawing import Drawing
from src.processing import get_augment_trans
from src.svg_extraction import get_drawing_paths
import clip
from src.utils import get_nouns
import torch
import pydiffvg
import copy
import numpy as np

pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')


class Cicada:
    def __init__(
        self,
        device,
        canvas_w=224,
        canvas_h=224,
        drawing_area={'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1},
        normalize_clip=True,
        max_width=40,
    ):
        self.device = device
        self.model, preprocess = clip.load('ViT-B/32', self.device, jit=False)
        self.clipConvLoss = CLIPConvLoss2(self.device)
        self.drawing = Drawing(canvas_w, canvas_h)
        self.augment_trans = get_augment_trans(
            self.drawing.canvas_width, normalize_clip
        )
        self.drawing_area = drawing_area
        self.attention_regions = []
        self.max_width = max_width
        self.t = 0

    def process_text(self, prompt, neg_prompts=["Written words.", "Text."]):
        self.nouns, noun_prompts = get_nouns()
        text_input = clip.tokenize(prompt).to(self.device)
        self.neg_text_features = []
        with torch.no_grad():
            self.nouns_features = self.model.encode_text(
                torch.cat([clip.tokenize(noun_prompts).to(self.device)])
            )
            self.text_features = self.model.encode_text(text_input)
            self.neg_text_features = [
                self.model.encode_text(
                    torch.cat([clip.tokenize(prompt).to(self.device)])
                )
                for prompt in neg_prompts
            ]

    def load_svg_shapes(self, svg_path):
        '''
        This will discard all existing traces and load those in an svg file
        ---
        input:
            svg_path: String;
        '''
        path_list = get_drawing_paths(svg_path)
        self.drawing.add_paths(path_list)

    def load_listed_shapes(self, shapes, shape_groups, fix=True):
        '''
        This will NOT discard existing shapes
        ---
        input:
            shapes: Path[];
            shape_groups: PathGroup[];
            fix: Bool;
        '''
        self.drawing.add_shapes(shapes, shape_groups, fix)

    def add_traces(self, trace_list, replace=False, fix_all=False):
        '''
        Add traces on
        ---
        input:
            trace_list: Trace[];
            replace: Bool;
            fix_all: Bool;
        '''
        if fix_all:
            for trace in trace_list:
                trace.is_fixed = True
        if replace:
            self.drawing.replace_traces(trace_list)
        else:
            self.drawing.add_traces(trace_list)
        self.initialize_variables()

    def repalce_traces(self, trace_list, fix_all=False):
        '''
        Replace traces in the same positions
        ---
        input:
            trace_list: Trace[];
            fix_all: Bool;
        '''
        if fix_all:
            for trace in trace_list:
                trace.is_fixed = True
        self.drawing.replace_traces(trace_list)
        self.initialize_variables()

    def add_random_shapes(self, num_rnd_traces):
        '''
        This will NOT discard existing shapes
        ---
        input:
            num_rnd_traces: Int;
        '''
        shapes, shape_groups = treebranch_initialization(
            self.drawing,
            num_rnd_traces,
            self.drawing_area,
        )
        self.drawing.add_shapes(shapes, shape_groups, fixed=False)

    def remove_traces(self, idx_list):
        '''
        Remove the traces indexed in idx_list
        ---
        input:
            num_rnd_traces: Int[];
        '''
        self.drawing.remove_traces(idx_list)
        self.initialize_variables()

    def initialize_variables(self):
        self.points_vars = []
        self.stroke_width_vars = []
        self.color_vars = []
        for trace in self.drawing.traces:
            trace.shape.points.requires_grad = True
            self.points_vars.append(trace.shape.points)
            trace.shape.stroke_width.requires_grad = True
            self.stroke_width_vars.append(trace.shape.stroke_width)
            trace.shape_group.stroke_color.requires_grad = True
            self.color_vars.append(trace.shape_group.stroke_color)

        self.render = pydiffvg.RenderFunction.apply
        self.mask = utils.area_mask(
            self.drawing.canvas_width, self.drawing.canvas_height, self.drawing_area
        ).to(self.device)
        self.points_vars0 = copy.deepcopy(self.points_vars)
        self.stroke_width_vars0 = copy.deepcopy(self.stroke_width_vars)
        self.color_vars0 = copy.deepcopy(self.color_vars)
        for k in range(len(self.color_vars0)):
            self.points_vars0[k].requires_grad = False
            self.stroke_width_vars0[k].requires_grad = False
            self.color_vars0[k].requires_grad = False
        self.img0 = copy.copy(self.drawing.img.detach())

    def initialize_optimizer(self, weight=1):
        self.points_optim = torch.optim.Adam(self.points_vars, lr=weight * 0.3)
        self.width_optim = torch.optim.Adam(self.stroke_width_vars, lr=weight * 0.3)
        self.color_optim = torch.optim.Adam(self.color_vars, lr=weight * 0.03)

    def build_img(self, t, shapes=None, shape_groups=None):
        if not shapes:
            shapes = [trace.shape for trace in self.drawing.traces]
            shape_groups = [trace.shape_group for trace in self.drawing.traces]
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.drawing.canvas_width, self.drawing.canvas_height, shapes, shape_groups
        )
        img = self.render(
            self.drawing.canvas_width,
            self.drawing.canvas_height,
            2,
            2,
            self.t,
            None,
            *scene_args,
        )
        self.t += 1
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img

    def set_penalizers(
        self,
        w_points=0.001,
        w_colors=0.01,
        w_widths=0.001,
        w_img=0.0,
        w_geo=3.5,
        w_global=1,
    ):
        self.w_points = w_points
        self.w_colors = w_colors
        self.w_widths = w_widths
        self.w_img = w_img
        self.w_geo = w_geo
        self.w_global = w_global

    def mutate_lr(self, increase_rate=5, num_iter=10):
        self.initialize_optimizer(increase_rate)
        for iter in range(num_iter):
            self.run_epoch()
        self.initialize_optimizer(1)

    @torch.no_grad()
    def mutate_respawn_traces(self, rate=0.3, num_sets=10, num_augs=4):
        unfixed_inds = [
            k
            for k in range(len(self.drawing.traces))
            if not self.drawing.traces[k].is_fixed
        ]
        N = round(rate * len(unfixed_inds))
        index_lists = [np.random.choice(unfixed_inds, N) for k in range(num_sets)]
        losses = []
        for idx_list in index_lists:
            shapes, shape_groups = self.drawing.all_shapes_except(idx_list)
            img = self.build_img(5, shapes, shape_groups)
            img_augs = []
            for n in range(num_augs):
                img_augs.append(self.augment_trans(img))
            im_batch = torch.cat(img_augs)
            img_features = self.model.encode_image(im_batch)
            loss = 0
            for n in range(num_augs):
                loss -= torch.cosine_similarity(
                    self.text_features, img_features[n : n + 1], dim=1
                )
            losses.append(loss.cpu().item())

        min_idx = np.argmin(losses)
        self.drawing.remove_traces(index_lists[min_idx])
        self.add_random_shapes(N)

    def run_epoch(self, t="deprecated", num_augs=4):
        self.points_optim.zero_grad()
        self.width_optim.zero_grad()
        self.color_optim.zero_grad()

        img = self.build_img(t)

        img_loss = (
            torch.norm((img - self.img0) * self.mask)
            if self.w_img > 0
            else torch.tensor(0, device=self.device)
        )

        self.img = img.cpu().permute(0, 2, 3, 1).squeeze(0)

        loss = 0

        img_augs = []
        for n in range(num_augs):
            img_augs.append(self.augment_trans(img))

        im_batch = torch.cat(img_augs)
        img_features = self.model.encode_image(im_batch)
        for n in range(num_augs):
            loss -= torch.cosine_similarity(
                self.text_features, img_features[n : n + 1], dim=1
            )
            for neg_text_feat in self.neg_text_features:
                loss += (
                    torch.cosine_similarity(
                        neg_text_feat, img_features[n : n + 1], dim=1
                    )
                    * 0.3
                )

        for att_region in self.attention_regions:
            cropped_batch = []
            cropped_img = img * att_region['mask'] + 1 - att_region['mask']
            for n in range(num_augs):
                cropped_batch.append(self.augment_trans(cropped_img))

            cropped_batch = torch.cat(cropped_batch)
            cropped_features = self.model.encode_image(cropped_batch)
            for n in range(num_augs):
                loss -= torch.cosine_similarity(
                    att_region['text_features'], cropped_features[n : n + 1], dim=1
                )

        self.img_features = img_features

        points_loss = 0
        widths_loss = 0
        colors_loss = 0

        for k in range(len(self.points_vars)):
            if self.drawing.traces[k].is_fixed:
                points_loss += torch.norm(self.points_vars[k] - self.points_vars0[k])
                colors_loss += torch.norm(self.color_vars[k] - self.color_vars0[k])
                widths_loss += torch.norm(
                    self.stroke_width_vars[k] - self.stroke_width_vars0[k]
                )

        loss += self.w_points * points_loss
        loss += self.w_colors * colors_loss
        loss += self.w_widths * widths_loss
        loss += self.w_img * img_loss

        geo_loss = self.clipConvLoss(img * self.mask + 1 - self.mask, self.img0)

        for l_name in geo_loss:
            loss += self.w_geo * geo_loss[l_name]

        # Backpropagate the gradients.
        loss = self.w_global * loss
        loss.backward()

        # Take a gradient descent step.
        self.points_optim.step()
        self.width_optim.step()
        self.color_optim.step()
        for trace in self.drawing.traces:
            trace.shape.stroke_width.data.clamp_(1.0, self.max_width)
            trace.shape_group.stroke_color.data.clamp_(0.0, 1.0)

        self.losses = {
            'global': loss,
            'points': points_loss,
            'widhts': widths_loss,
            'colors': colors_loss,
            'image': img_loss,
            'geometric': geo_loss,
        }

    def prune(self, prune_ratio, num_augs=4):
        with torch.no_grad():
            # Get points of tied traces
            fixed_points = []
            for trace in self.drawing.traces:
                if trace.is_fixed:
                    fixed_points += [
                        x.unsqueeze(0)
                        for i, x in enumerate(trace.shape.points)
                        if i % 3 == 0
                    ]  # only points the path goes through

            # Compute distances
            dists = []
            if fixed_points:
                fixed_points = torch.cat(fixed_points, 0)
                for trace in self.drawing.traces:
                    if trace.is_fixed:
                        dists.append(-1000)  # We don't remove fixed traces
                    else:
                        points = [
                            x.unsqueeze(0)
                            for i, x in enumerate(trace.shape.points)
                            if i % 3 == 0
                        ]  # only points the path goes through
                        min_dists = []
                        for point in points:
                            d = torch.norm(point - fixed_points, dim=1)
                            d = min(d)
                            min_dists.append(d.item())

                        dists.append(min(min_dists))

            # Compute losses
            losses = []
            for n, trace in enumerate(self.drawing.traces):
                if trace.is_fixed:
                    losses.append(1000)  # We don't remove fixed traces
                else:
                    # Compute the loss if we take out the k-th path
                    shapes, shape_groups = self.drawing.all_shapes_but_kth(n)
                    img = self.build_img(5, shapes, shape_groups)
                    img_augs = []
                    for n in range(num_augs):
                        img_augs.append(self.augment_trans(img))
                    im_batch = torch.cat(img_augs)
                    img_features = self.model.encode_image(im_batch)
                    loss = 0
                    for n in range(num_augs):
                        loss -= torch.cosine_similarity(
                            self.text_features, img_features[n : n + 1], dim=1
                        )
                    losses.append(loss.cpu().item())

            # Compute scores
            scores = [-0.01 * dists[k] ** (0.5) + losses[k] for k in range(len(losses))]

            # Actual pruning
            inds = utils.k_min_elements(
                scores, int(prune_ratio * len(self.drawing.traces))
            )
            self.drawing.remove_traces(inds)

        self.initialize_variables()
