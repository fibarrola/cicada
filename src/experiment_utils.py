import torch
from src import utils
from src.render_design import add_shape_groups


def get_fixed_paths(drawing_model, args, n_keep):
    with torch.no_grad():
        drawn_points = []
        for k in range(drawing_model.num_sketch_paths):
            drawn_points += [
                x.unsqueeze(0)
                for i, x in enumerate(drawing_model.shapes[k].points)
                if i % 3 == 0
            ]
        drawn_points = (
            torch.cat(drawn_points, 0) if drawing_model.num_sketch_paths > 0 else []
        )

        losses = []
        dists = []
        for k in range(
            drawing_model.num_sketch_paths, len(drawing_model.stroke_width_vars)
        ):

            # Compute the distance between the set of user's partial sketch points and random curve points
            if len(drawn_points) > 0:
                points = [
                    x.unsqueeze(0)
                    for i, x in enumerate(drawing_model.shapes[k].points)
                    if i % 3 == 0
                ]  # only points the path goes through
                min_dists = []
                for point in points:
                    d = torch.norm(point - drawn_points, dim=1)
                    d = min(d)
                    min_dists.append(d.item())

                dists.append(min(min_dists))

            # Compute the loss if we take out the k-th path
            shapes = drawing_model.shapes[:k] + drawing_model.shapes[k + 1 :]
            shape_groups = add_shape_groups(
                drawing_model.shape_groups[:k], drawing_model.shape_groups[k + 1 :]
            )
            img = drawing_model.build_img(shapes, shape_groups, 5)
            img_augs = []
            for n in range(args.num_augs):
                img_augs.append(drawing_model.augment_trans(img))
            im_batch = torch.cat(img_augs)
            img_features = drawing_model.model.encode_image(im_batch)
            loss = 0
            for n in range(args.num_augs):
                loss -= torch.cosine_similarity(
                    drawing_model.text_features, img_features[n : n + 1], dim=1
                )
            losses.append(loss.cpu().item())

        scores = (
            [-losses[k] for k in range(len(losses))]
            if len(drawn_points) > 0
            else losses
        )
        inds = utils.k_min_elements(scores, n_keep)

        extra_shapes = [
            drawing_model.shapes[idx + drawing_model.num_sketch_paths] for idx in inds
        ]
        extra_shape_groups = [
            drawing_model.shape_groups[idx + drawing_model.num_sketch_paths]
            for idx in inds
        ]

        shapes = drawing_model.user_sketch.shapes + extra_shapes
        shape_groups = add_shape_groups(
            drawing_model.user_sketch.shape_groups, extra_shape_groups
        )
        fixed_inds = list(range(len(drawing_model.user_sketch.shapes), len(shapes)))

        for s in range(len(shapes)):
            shapes[s].points.requires_grad = False
            shapes[s].stroke_width.requires_grad = False
            shape_groups[s].stroke_color.requires_grad = False

    return shapes, shape_groups, fixed_inds
