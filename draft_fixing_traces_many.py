from cgitb import text
from concurrent.futures import process
from contextlib import AsyncExitStack
import torch
import pydiffvg
from config import args
from pathlib import Path
from src import utils
from drawing_model import DrawingModel
import numpy as np
from src.fid_score import main_within, get_statistics

NUM_TRIALS = 20
GENS_PER_TRIAL = 20

args.num_iter = 1000
textfile = open('results/fixing_traces.txt', 'w')


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


names = ['chair', 'hat', 'lamp', 'pot']
yy0 = [0.5, 0.6, 0.0, 0.0]
yy1 = [1.0, 1.0, 0.5, 0.5]
prompts = [
    'A red chair.',
    'A drawing of a hat.',
    'A drawing of a lamp.',
    'A drawing of a pot.',
]


for n, name in enumerate(names):

    args.svg_path = f"data/drawing_{name}.svg"
    args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': yy0[n], 'y1': yy1[n]}
    args.clip_prompt = f"A drawing of a {name}"

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    save_path = Path("results/").joinpath(f'fix_paths/{name}')
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = str(save_path) + '/'

    for trial in range(NUM_TRIALS):

        drawing_model = DrawingModel(args, device)
        drawing_model.process_text(args)
        drawing_model.initialize_shapes(args)
        drawing_model.initialize_variables(args)
        drawing_model.initialize_optimizer()

        t0 = 0
        for t in range(args.num_iter):

            if t + 1 == args.num_iter:
                with torch.no_grad():
                    pydiffvg.imwrite(
                        drawing_model.img,
                        save_path + f'from_scratch/{trial+200}.png',
                        gamma=1,
                    )

            drawing_model.run_epoch(t, args)

            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model.losses['global'].item()
            )

        continue

        if trial < 3:
            shapes, shape_groups, fixed_inds = drawing_model.get_fixed_paths(args, 6)
            for gen in range(GENS_PER_TRIAL):
                drawing_model = DrawingModel(args, device)
                drawing_model.process_text(args)
                drawing_model.load_shapes(args, shapes, shape_groups, fixed_inds)
                drawing_model.initialize_variables(args)
                new_mask = 1 - torch.floor(drawing_model.img0)
                drawing_model.mask = torch.ceil((drawing_model.mask + new_mask) / 2)
                drawing_model.initialize_optimizer()
                with torch.no_grad():
                    pydiffvg.imwrite(
                        drawing_model.img0.squeeze(0).permute(1, 2, 0).cpu(),
                        save_path + f'fixe_lines/{trial+100}.png',
                        gamma=1,
                    )
                    pydiffvg.imwrite(
                        drawing_model.mask.detach().squeeze(0).permute(1, 2, 0).cpu(),
                        save_path + f'fixe_lines/{trial+100}.png',
                        gamma=1,
                    )
                for t in range(args.num_iter):

                    drawing_model.run_epoch(t, args)

                    if t + 1 == args.num_iter:
                        with torch.no_grad():
                            pydiffvg.imwrite(
                                drawing_model.img,
                                save_path + f'gen_trial{trial+100}/{gen}.png',
                                gamma=1,
                            )

                    utils.printProgressBar(
                        t + 1, args.num_iter, drawing_model.losses['global'].item()
                    )

# for n, name in enumerate(names):
#     for process_name in ['from_scratch', 'gen_trial0', 'gen_trial1', 'gen_trial100', 'gen_trial101', 'gen_trial102']:

#         mu, S = get_statistics(f"results/fix_paths/{name}/{process_name}")
#         fids = []
#         for it in range(5):
#             fids.append(main_within(f"results/fix_paths/{name}/{process_name}"))
#         textfile.write(f"object: {name} \n")
#         textfile.write(f"process: {process_name} \n")
#         textfile.write(f" mean= {np.mean(fids)}, stdev= {np.std(fids)} \n")
#         textfile.write(f"cov norm = {np.linalg.norm(S)} \n")
#         textfile.write("\n")

# textfile.close()

import pandas as pd
import src.fid_score as fid
import plotly.express as px

fid_data = []
fid_data2 = []
for n, name in enumerate(names):
    for process_name in [
        'from_scratch',
        'gen_trial0',
        'gen_trial1',
        'gen_trial100',
        'gen_trial101',
        'gen_trial102',
    ]:
        mu, S = get_statistics(f"results/fix_paths/{name}/{process_name}")
        gen_type = 'standard' if process_name == 'from_scratch' else 'trace-conditioned'
        fid_data.append(
            {
                'Covariance Norm': np.linalg.norm(S),
                'name': name,
                'generation': gen_type,
            }
        )

import pickle

with open('results/fix_paths/data.pkl', 'wb') as f:
    pickle.dump(fid_data, f)

df = pd.DataFrame(fid_data)

fig = px.scatter(
    df,
    x="name",
    y="Covariance Norm",
    color="generation",  # size=[2 for x in range(len(df))]
)
fig.show()
