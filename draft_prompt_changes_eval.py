import torch
import pydiffvg
from config import args
from pathlib import Path
from src import utils
from drawing_model import DrawingModel
import src.fid_score as fid
import copy
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
from src.style import image_loader


device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

NUM_TRIALS = 10
NUM_SETS = 5
NUM_STD = 30
args.num_iter = 1000

names = ['chair', 'hat', 'lamp', 'pot', 'boat', 'dress', 'shoe', 'bust']
yy0 = [0.5, 0.6, 0.0, 0.0, 0.35, 0.0, 0.0, 0.5]
yy1 = [1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0]
prompts_A = [
    'A tall red chair.',
    'A drawing of a pointy black hat.',
    'A drawing of a tall green lamp.',
    'A drawing of a shallow pot.',
    'A drawing of a sail boat.',
    'A blue and red short dress.',
    'A high-heel brown shoe.',
    'A bust of a roman emperor.',
]
prompts_B = [
    'A short blue chair.',
    'A drawing of a flat pink hat.',
    'A drawing of a round black lamp.',
    'A drawing of a large pot.',
    'A drawing of a steam boat.',
    'A blue and green long dress.',
    'A high-heel black shoe.',
    'A bust of an army general.',
]

# names = names[4:]
# yy0 = yy0[4:]
# yy1 = yy1[4:]
# prompts_A = prompts_A[4:]
# prompts_B = prompts_B[4:]

# for n, name in enumerate(names):

#     args.svg_path = f"data/drawing_{name}.svg"
#     args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': yy0[n], 'y1': yy1[n]}

#     for trial_set in range(NUM_SETS):

#         save_path = Path("results/").joinpath(f'prompt_change/{name}/pA/')
#         save_path.mkdir(parents=True, exist_ok=True)
#         save_path = Path("results/").joinpath(f'prompt_change/{name}/pAB_{trial_set}/')
#         save_path.mkdir(parents=True, exist_ok=True)
#         save_path = Path("results/").joinpath(f'prompt_change/{name}/pB/')
#         save_path.mkdir(parents=True, exist_ok=True)

#         args.clip_prompt = prompts_A[n]

#         ###############################
#         # Standard with prompt A ######
#         ###############################
#         drawing_model_A = DrawingModel(args, device)
#         drawing_model_A.process_text(args)
#         drawing_model_A.load_svg_shapes(args.svg_path)
#         drawing_model_A.add_random_shapes(args.num_paths, args)
#         drawing_model_A.initialize_variables()
#         drawing_model_A.initialize_optimizer()

#         for t in range(args.num_iter):

#             drawing_model_A.run_epoch(t, args)
#             utils.printProgressBar(
#                 t + 1, args.num_iter, drawing_model_A.losses['global'].item()
#             )

#         with torch.no_grad():
#             pydiffvg.imwrite(
#                 drawing_model_A.img,
#                 f'results/prompt_change/{name}/pA/{trial_set}.png',
#                 gamma=1,
#             )

#         ###############################
#         # Starting from drawing A #####
#         ###############################

#         for trial in range(NUM_TRIALS):
#             # drawing_model_AB = copy.deepcopy(drawing_model_A)
#             drawing_model_AB = DrawingModel(args, device)
#             args.clip_prompt = prompts_B[n]
#             drawing_model_AB.process_text(args)
#             drawing_model_AB.load_svg_shapes(args.svg_path)
#             N = len(drawing_model_AB.shapes)
#             drawing_model_AB.load_listed_shapes(args, drawing_model_A.shapes[N:], drawing_model_A.shape_groups[N:], tie=False)
#             # drawing_model_AB.shapes = copy.deepcopy(drawing_model_A.shapes)
#             # drawing_model_AB.shape_groups = copy.deepcopy(drawing_model_A.shape_groups)
#             # drawing_model_AB.num_sketch_paths = copy.deepcopy(
#             #     drawing_model_A.num_sketch_paths
#             # )
#             # drawing_model_AB.augment_trans = copy.deepcopy(
#             #     drawing_model_A.augment_trans
#             # )
#             # drawing_model_AB.user_sketch = copy.deepcopy(drawing_model_A.user_sketch)
#             drawing_model_AB.initialize_variables()
#             drawing_model_AB.initialize_optimizer()

#             for t in range(args.num_iter):
#                 drawing_model_AB.run_epoch(t, args)
#                 utils.printProgressBar(
#                     t + 1, args.num_iter, drawing_model_AB.losses['global'].item()
#                 )

#             with torch.no_grad():
#                 pydiffvg.imwrite(
#                     drawing_model_AB.img,
#                     f'results/prompt_change/{name}/pAB_{trial_set}/{trial}.png',
#                     gamma=1,
#                 )

#     ###############################
#     # Standard with prompt B ######
#     ###############################
#     for trial in range(NUM_STD):

#         drawing_model_B = DrawingModel(args, device)
#         drawing_model_B.process_text(args)
#         drawing_model_B.load_svg_shapes(args.svg_path)
#         drawing_model_B.add_random_shapes(args.num_paths, args)
#         drawing_model_B.initialize_variables()
#         drawing_model_B.initialize_optimizer()

#         for t in range(args.num_iter):
#             drawing_model_B.run_epoch(t, args)
#             utils.printProgressBar(
#                 t + 1, args.num_iter, drawing_model_B.losses['global'].item()
#             )

#         with torch.no_grad():
#             pydiffvg.imwrite(
#                 drawing_model_B.img,
#                 f'results/prompt_change/{name}/pB/{trial}.png',
#                 gamma=1,
#             )

device = "cuda:0" if torch.cuda.is_available() else "cpu"

cov_data = []
loss_data = []
# print(['pB' for ts in range(NUM_SETS)] + [f'pAB_{ts}' for ts in range(NUM_SETS)])
for n, name in enumerate(names):

    args.clip_prompt = prompts_B[n]
    drawing_model = DrawingModel(args, device)
    drawing_model.process_text(args)

    for process_name in ['pB' for ts in range(NUM_SETS)] + [
        f'pAB_{ts}' for ts in range(NUM_SETS)
    ]:
        subset_dim = 10 if process_name == 'pB' else None
        mu, S = fid.get_statistics(
            f"results/prompt_change/{name}/{process_name}",
            rand_sampled_set_dim=subset_dim,
        )
        gen_type = 'standard' if process_name == 'pB' else 'prompt-change-conditioned'
        cov_data.append(
            {
                'Covariance Norm': np.linalg.norm(S),
                'name': name,
                'generation': gen_type,
            }
        )

        filenames = os.listdir(f"results/prompt_change/{name}/{process_name}")
        for filename in filenames:
            img = image_loader(
                f"results/prompt_change/{name}/{process_name}/{filename}"
            )
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
            loss_data.append(
                {
                    'loss': loss.detach().cpu().item(),
                    'name': name,
                    'generation': gen_type,
                }
            )


# with open('results/prompt_change/data_02.pkl', 'wb') as f:
#     pickle.dump(cov_data, f)

df = pd.DataFrame(cov_data)

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


fig = go.Figure()


xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['Covariance Norm']
fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color='rgba(255,157,0,1)'))

xx = df.query('generation=="prompt-change-conditioned"')['name']
yy = df.query('generation=="prompt-change-conditioned"')['Covariance Norm']
fig.add_trace(
    go.Box(
        y=yy, x=xx, name='prompt-change-conditioned', marker_color='rgba(0,83,170,1)'
    )
)

fig.update_layout(
    boxmode='group',
    yaxis_title="Norm of Covariance Matrix",
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
    font={'size': 16},
)
fig.show()

# NOw FOR LOSSES
df = pd.DataFrame(loss_data)
fig = go.Figure()

xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['loss']

fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color='rgba(255,157,0,1)'))

xx = df.query('generation=="prompt-change-conditioned"')['name']
yy = df.query('generation=="prompt-change-conditioned"')['loss']

fig.add_trace(
    go.Box(
        y=yy, x=xx, name='prompt-change-conditioned', marker_color='rgba(0,83,170,1)'
    )
)

fig.update_layout(
    boxmode='group',
    yaxis_title="Semantic Loss",
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
    font={'size': 16},
)
fig.show()

df = pd.DataFrame(loss_data)
fig = go.Figure()

xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['loss']

fig.add_trace(
    go.Box(
        y=yy,
        x=xx,
        name='standard',
        marker_color='rgba(255,157,0,1)',
        quartilemethod="inclusive",
    )
)

xx = df.query('generation=="prompt-change-conditioned"')['name']
yy = df.query('generation=="prompt-change-conditioned"')['loss']

fig.add_trace(
    go.Box(
        y=yy,
        x=xx,
        name='prompt-change-conditioned',
        marker_color='rgba(0,83,170,1)',
        quartilemethod="inclusive",
    )
)

fig.update_layout(
    boxmode='group',
    yaxis_title="Semantic Loss",
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
    font={'size': 16},
)
fig.show()

df = pd.DataFrame(loss_data)
fig = go.Figure()

xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['loss']

fig.add_trace(
    go.Box(
        y=yy,
        x=xx,
        name='standard',
        marker_color='rgba(255,157,0,1)',
        quartilemethod="exclusive",
    )
)

xx = df.query('generation=="prompt-change-conditioned"')['name']
yy = df.query('generation=="prompt-change-conditioned"')['loss']

fig.add_trace(
    go.Box(
        y=yy,
        x=xx,
        name='prompt-change-conditioned',
        marker_color='rgba(0,83,170,1)',
        quartilemethod="exclusive",
    )
)

fig.update_layout(
    boxmode='group',
    yaxis_title="Semantic Loss",
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
    font={'size': 16},
)
fig.show()
