import torch
from experiments.penalization_effect import CREATE_SAMPLES
import pydiffvg
from src.config import args
from pathlib import Path
from src import utils
from src.drawing_model import Cicada
import src.fid_score as fid
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go


device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

NUM_TRIALS = 2  # 10
NUM_SETS = 1  # 5
NUM_STD = 1  # 30
SAVE_PATH = "prompt_changes5"
CREATE_SAMPLES = False
args.num_iter = 1000

names = ['chair', 'hat', 'lamp', 'pot', 'boat', 'dress', 'shoe', 'bust']
yy0 = [0.5, 0.6, 0.0, 0.0, 0.35, 0.0, 0.0, 0.5]
yy1 = [1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.55, 1.0]
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


if CREATE_SAMPLES:
    for n, name in enumerate(names):

        args.svg_path = f"data/drawing_{name}.svg"
        args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': yy0[n], 'y1': yy1[n]}

        for trial_set in range(NUM_SETS):

            save_path = Path("results/").joinpath(f'{SAVE_PATH}/{name}/pA/')
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = Path("results/").joinpath(f'{SAVE_PATH}/{name}/pAB_{trial_set}/')
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = Path("results/").joinpath(f'{SAVE_PATH}/{name}/pB/')
            save_path.mkdir(parents=True, exist_ok=True)

            args.clip_prompt = prompts_A[n]

            ###############################
            # Standard with prompt A ######
            ###############################
            cicada_A = Cicada(args, device)
            cicada_A.process_text(args)
            cicada_A.load_svg_shapes(args.svg_path)
            cicada_A.add_random_shapes(args.num_paths)
            cicada_A.initialize_variables()
            cicada_A.initialize_optimizer()

            for t in range(args.num_iter):

                cicada_A.run_epoch(t, args)
                utils.printProgressBar(
                    t + 1, args.num_iter, cicada_A.losses['global'].item()
                )

            with torch.no_grad():
                pydiffvg.imwrite(
                    cicada_A.img,
                    f'results/{SAVE_PATH}/{name}/pA/{trial_set}.png',
                    gamma=1,
                )

            ###############################
            # Starting from drawing A #####
            ###############################

            for trial in range(NUM_TRIALS):
                cicada_AB = Cicada(args, device)
                args.clip_prompt = prompts_B[n]
                cicada_AB.process_text(args)
                cicada_AB.load_svg_shapes(args.svg_path)
                N = len(cicada_AB.shapes)
                cicada_AB.load_listed_shapes(
                    cicada_A.shapes[N:],
                    cicada_A.shape_groups[N:],
                    fix=False,
                )
                cicada_AB.initialize_variables()
                cicada_AB.initialize_optimizer()

                for t in range(args.num_iter):
                    cicada_AB.run_epoch(t, args)
                    utils.printProgressBar(
                        t + 1, args.num_iter, cicada_AB.losses['global'].item()
                    )

                with torch.no_grad():
                    pydiffvg.imwrite(
                        cicada_AB.img,
                        f'results/{SAVE_PATH}/{name}/pAB_{trial_set}/{trial}.png',
                        gamma=1,
                    )

        ###############################
        # Standard with prompt B ######
        ###############################
        for trial in range(NUM_STD):

            cicada_B = Cicada(args, device)
            cicada_B.process_text(args)
            cicada_B.load_svg_shapes(args.svg_path)
            cicada_B.add_random_shapes(args.num_paths)
            cicada_B.initialize_variables()
            cicada_B.initialize_optimizer()

            for t in range(args.num_iter):
                cicada_B.run_epoch(t, args)
                utils.printProgressBar(
                    t + 1, args.num_iter, cicada_B.losses['global'].item()
                )

            with torch.no_grad():
                pydiffvg.imwrite(
                    cicada_B.img,
                    f'results/{SAVE_PATH}/{name}/pB/{trial}.png',
                    gamma=1,
                )


fid_data = []
print(['pB' for ts in range(NUM_SETS)] + [f'pAB_{ts}' for ts in range(NUM_SETS)])
for n, name in enumerate(names):
    for process_name in ['pB' for ts in range(NUM_SETS)] + [
        f'pAB_{ts}' for ts in range(NUM_SETS)
    ]:
        subset_dim = 10 if process_name == 'pB' else None
        mu, S = fid.get_statistics(
            f"results/{SAVE_PATH}/{name}/{process_name}",
            rand_sampled_set_dim=subset_dim,
        )
        gen_type = 'standard' if process_name == 'pB' else 'prompt-change-conditioned'
        fid_data.append(
            {
                'Entropy': np.linalg.norm(S),
                'name': name,
                'generation': gen_type,
            }
        )


with open(f'results/{SAVE_PATH}/data_02.pkl', 'wb') as f:
    pickle.dump(fid_data, f)

df = pd.DataFrame(fid_data)

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
yy = df.query('generation=="standard"')['Entropy']
fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color='rgba(255,157,0,1)'))

xx = df.query('generation=="prompt-change-conditioned"')['name']
yy = df.query('generation=="prompt-change-conditioned"')['Entropy']
fig.add_trace(
    go.Box(
        y=yy, x=xx, name='prompt-change-conditioned', marker_color='rgba(0,83,170,1)'
    )
)

fig.update_layout(
    boxmode='group',
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
)
fig.show()
