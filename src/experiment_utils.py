import torch
from src import utils

def get_fixed_inds(drawing_model, n_keep):
    
    with torch.no_grad():

        # Get points of tied traces
        fixed_points = []
        for trace in drawing_model.drawing.traces:
            if trace.is_fixed:
                fixed_points += [
                    x.unsqueeze(0)
                    for i, x in enumerate(trace.shape.points)
                    if i % 3 == 0
                ]  # only points the path goes through

        # Compute losses
        losses = []
        for n, trace in enumerate(drawing_model.drawing.traces):
            if trace.is_fixed:
                losses.append(-1000)  # We don't remove fixed traces
            else:
                # Compute the loss if we take out the k-th path
                shapes, shape_groups = drawing_model.drawing.all_shapes_but_kth(n)
                img = drawing_model.build_img(5, shapes, shape_groups)
                img_augs = []
                for n in range(drawing_model.num_augs):
                    img_augs.append(drawing_model.augment_trans(img))
                im_batch = torch.cat(img_augs)
                img_features = drawing_model.model.encode_image(im_batch)
                loss = 0
                for n in range(drawing_model.num_augs):
                    loss -= torch.cosine_similarity(
                        drawing_model.text_features, img_features[n : n + 1], dim=1
                    )
                losses.append(loss.cpu().item())

        # Compute scores
        inds = utils.k_max_elements(losses, n_keep)

    return inds


def get_fixed_paths(drawing_model, n_keep):

    with torch.no_grad():

        # Get points of tied traces
        fixed_points = []
        for trace in drawing_model.drawing.traces:
            if trace.is_fixed:
                fixed_points += [
                    x.unsqueeze(0)
                    for i, x in enumerate(trace.shape.points)
                    if i % 3 == 0
                ]  # only points the path goes through

        # Compute losses
        losses = []
        for n, trace in enumerate(drawing_model.drawing.traces):
            if trace.is_fixed:
                losses.append(-1000)  # We don't remove fixed traces
            else:
                # Compute the loss if we take out the k-th path
                shapes, shape_groups = drawing_model.drawing.all_shapes_but_kth(n)
                img = drawing_model.build_img(5, shapes, shape_groups)
                img_augs = []
                for n in range(drawing_model.num_augs):
                    img_augs.append(drawing_model.augment_trans(img))
                im_batch = torch.cat(img_augs)
                img_features = drawing_model.model.encode_image(im_batch)
                loss = 0
                for n in range(drawing_model.num_augs):
                    loss -= torch.cosine_similarity(
                        drawing_model.text_features, img_features[n : n + 1], dim=1
                    )
                losses.append(loss.cpu().item())

        # Compute scores
        inds = utils.k_max_elements(losses, n_keep)

        shapes = []
        shape_groups = []
        for i, trace in enumerate(drawing_model.drawing.traces):
            if trace.is_fixed or i in inds:
                shapes.append(trace.shape)
                shape_groups.append(trace.shape_group)

    return shapes, shape_groups


