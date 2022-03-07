import pydiffvg
import torch
import pickle

def render_save_img(path_list, canvas_height, canvas_width):

    # Initialize Curves
    shapes = []
    shape_groups = []

    # First the ones from my drawing
    for dpath in path_list:
        num_control_points = torch.zeros(dpath.num_segments, dtype = torch.int32) + 2
        points = torch.zeros_like(dpath.path)
        stroke_width = dpath.width*100
        points[:, 0] = canvas_width*dpath.path[:,0]
        points[:, 1] = canvas_height*dpath.path[:,1]
        path = pydiffvg.Path(num_control_points = num_control_points,
                            points = points, stroke_width = stroke_width,
                            is_closed = False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor(
            [len(shapes) - 1]), fill_color = None, stroke_color = dpath.color)
        shape_groups.append(path_group)

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups
    )
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

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
        
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
    
    with open('tmp/img0.pkl', 'wb') as f:
        pickle.dump(img, f)
    with open('tmp/points_vars.pkl', 'wb') as f:
        pickle.dump(points_vars, f)
    with open('tmp/stroke_width_vars.pkl', 'wb') as f:
        pickle.dump(stroke_width_vars, f)
    with open('tmp/color_vars.pkl', 'wb') as f:
        pickle.dump(color_vars, f)

    return shapes, shape_groups