from tokenize import group
import torch
import pydiffvg


class Trace:
    def __init__(self, shape, shape_group, is_fixed):
        self.shape = shape
        self.shape_group = shape_group
        self.is_fixed = is_fixed


class Drawing:
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.traces = []
        self.img = None

    def add_paths(self, path_list):
        shapes = []
        shape_groups = []
        for dpath in path_list:
            path = dpath.path.detach().clone()
            width = dpath.width.detach().clone()
            color = dpath.color.detach().clone()
            num_control_points = torch.zeros(dpath.num_segments, dtype=torch.int32) + 2
            points = torch.zeros_like(path)
            stroke_width = width * 100
            points[:, 0] = self.canvas_width * path[:, 0]
            points[:, 1] = self.canvas_height * path[:, 1]
            shape = pydiffvg.Path(
                num_control_points=num_control_points,
                points=points,
                stroke_width=stroke_width,
                is_closed=False,
            )
            shapes.append(shape)
            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color,
            )
            shape_groups.append(shape_group)
            self.traces.append(Trace(shape, shape_group, True))

        if not path_list:
            img = torch.ones(
                (1, 3, self.canvas_height, self.canvas_width),
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
            )
        else:
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                self.canvas_width, self.canvas_height, shapes, shape_groups
            )
            render = pydiffvg.RenderFunction.apply
            img = render(
                self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args
            )
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
                img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()
            ) * (1 - img[:, :, 3:4])
            img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)

        self.img = img

    def add_shapes(self, shapes, shape_groups, fixed):
        N = len(self.traces)
        print(N)
        for k in range(len(shapes)):
            group = shape_groups[k]
            shape_groups[k].shape_ids = torch.tensor([k + N])
            self.traces.append(Trace(shapes[k], group, fixed))
        
        shapes = [trace.shape for trace in self.traces]
        shape_groups = [trace.shape_group for trace in self.traces]

        
        scene_args = pydiffvg.RenderFunction.serialize_scene(
                self.canvas_width, self.canvas_height, shapes, shape_groups
            )
        render = pydiffvg.RenderFunction.apply
        img = render(
            self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args
        )
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)
        self.img = img

    def remove_traces(self, inds):
        self.traces = [self.traces[i] for i in range(len(self.traces)) if i not in inds]
        for k in range(len(self.traces)):
            self.traces[k].shape_group.shape_ids = torch.tensor([k])

    def all_shapes_but_kth(self, k):
        shapes = []
        shape_groups = []
        count = 0
        for n, trace in enumerate(self.traces):
            if n != k:
                shapes.append(trace.shape)
                group = trace.shape_group
                group.shape_ids = torch.tensor([count])
                shape_groups.append(group)
                count += 1
        return shapes, shape_groups
