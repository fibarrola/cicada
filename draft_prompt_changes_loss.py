import torch
import pydiffvg
from config import args
from pathlib import Path
from src import loss, utils
from drawing_model import DrawingModel
import src.fid_score as fid
import copy
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

NUM_SETS = 7
NUM_STD = 7
NUM_ITER = 750


names = ['chair', 'hat', 'lamp', 'pot']
yy0 = [0.5, 0.6, 0.0, 0.0]
yy1 = [1.0, 1.0, 0.5, 0.5]
prompts_A = [
    'A tall red chair.',
    'A drawing of a pointy black hat.',
    'A drawing of a tall green lamp.',
    'A drawing of a shallow pot.',
]
prompts_B = [
    'A short blue chair.',
    'A drawing of a flat pink hat.',
    'A drawing of a round black lamp.',
    'A drawing of a large pot.',
]

losses = {'name': [], 'iter': [], 'loss': [], 'type': [], 'trial': []}

for n, name in enumerate(names):

    args.svg_path = f"data/drawing_{name}.svg"
    args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': yy0[n], 'y1': yy1[n]}

    args.clip_prompt = prompts_B[n]
    aux_d_model = DrawingModel(args, device)
    aux_d_model.process_text(args)

    for trial_set in range(NUM_SETS):

        save_path = Path("results/").joinpath(f'prompt_change5/{name}/pA/')
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = Path("results/").joinpath(f'prompt_change5/{name}/pAB_{trial_set}/')
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = Path("results/").joinpath(f'prompt_change5/{name}/pB/')
        save_path.mkdir(parents=True, exist_ok=True)

        args.clip_prompt = prompts_A[n]
        args.num_iter = NUM_ITER

        ###############################
        # Standard with prompt A ######
        ###############################
        drawing_model_A = DrawingModel(args, device)
        drawing_model_A.process_text(args)
        drawing_model_A.load_svg_shapes(args)
        drawing_model_A.add_random_shapes(args.num_paths, args)
        drawing_model_A.initialize_variables(args)
        drawing_model_A.initialize_optimizer()

        for t in range(args.num_iter):

            drawing_model_A.run_epoch(t, args)
            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model_A.losses['global'].item()
            )
            with torch.no_grad():
                prompt_loss = utils.get_prompt_loss(
                    drawing_model_A.img_features, aux_d_model.text_features, args
                )
                losses['name'].append(name)
                losses['iter'].append(t)
                losses['type'].append('Old prompt')
                losses['loss'].append(prompt_loss.cpu().item())
                losses['trial'].append(trial_set + 3)

        # with torch.no_grad():
        #     pydiffvg.imwrite(
        #         drawing_model_A.img,
        #         f'results/prompt_change5/{name}/pA/{trial_set}.png',
        #         gamma=1,
        #     )

        ###############################
        # Starting from drawing A #####
        ###############################

        drawing_model_AB = DrawingModel(args, device)
        args.clip_prompt = prompts_B[n]
        drawing_model_AB.process_text(args)
        drawing_model_AB.shapes = copy.deepcopy(drawing_model_A.shapes)
        drawing_model_AB.shape_groups = copy.deepcopy(drawing_model_A.shape_groups)
        drawing_model_AB.num_sketch_paths = copy.deepcopy(
            drawing_model_A.num_sketch_paths
        )
        drawing_model_AB.augment_trans = copy.deepcopy(drawing_model_A.augment_trans)
        drawing_model_AB.user_sketch = copy.deepcopy(drawing_model_A.user_sketch)
        drawing_model_AB.fixed_inds = []
        drawing_model_AB.initialize_variables(args)
        drawing_model_AB.initialize_optimizer()

        for t in range(args.num_iter):
            drawing_model_AB.run_epoch(t, args)
            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model_AB.losses['global'].item()
            )
            with torch.no_grad():
                prompt_loss = utils.get_prompt_loss(
                    drawing_model_AB.img_features, aux_d_model.text_features, args
                )
                losses['name'].append(name)
                losses['iter'].append(t + args.num_iter)
                losses['type'].append('Changed prompt')
                losses['loss'].append(prompt_loss.cpu().item())
                losses['trial'].append(trial_set + 3)

        # with torch.no_grad():
        #     pydiffvg.imwrite(
        #         drawing_model_AB.img,
        #         f'results/prompt_change5/{name}/pAB_{trial_set}/{77}.png',
        #         gamma=1,
        #     )

    ###############################
    # Standard with prompt B ######
    ###############################
    args.num_iter = 2 * NUM_ITER
    for trial in range(NUM_STD):

        drawing_model_B = DrawingModel(args, device)
        drawing_model_B.process_text(args)
        drawing_model_B.load_svg_shapes(args)
        drawing_model_B.add_random_shapes(args.num_paths, args)
        drawing_model_B.initialize_variables(args)
        drawing_model_B.initialize_optimizer()

        for t in range(args.num_iter):
            drawing_model_B.run_epoch(t, args)
            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model_B.losses['global'].item()
            )
            with torch.no_grad():
                prompt_loss = utils.get_prompt_loss(
                    drawing_model_B.img_features, aux_d_model.text_features, args
                )
                losses['name'].append(name)
                losses['iter'].append(t)
                losses['type'].append('New prompt')
                losses['loss'].append(prompt_loss.cpu().item())
                losses['trial'].append(trial + 3)

        # with torch.no_grad():
        #     pydiffvg.imwrite(
        #         drawing_model_B.img,
        #         f'results/prompt_change5/{name}/pB/{trial}.png',
        #         gamma=1,
        #     )

# data = [{'iter': k+1, 'loss': losses[k].item() } for k in range(len(losses))]
df = pd.DataFrame(losses)
print(df)

with open('results/prompt_change5/losses2.pkl', 'wb') as f:
    pickle.dump(df, f)

fig = px.scatter(df, x='iter', y='loss', color='type')
fig.show()


# df = pd.DataFrame(fid_data)

# fig = px.scatter(
#     df,
#     x="name",
#     y="Covariance Norm",
#     color="generation",  # size=[2 for x in range(len(df))]
# )
# fig.show()

# fig = px.box(
#     df,
#     x="name",
#     y="Covariance Norm",
#     color="generation",  # size=[2 for x in range(len(df))]
# )
# fig.show()
