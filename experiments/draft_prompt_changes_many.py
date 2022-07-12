import torch
import pydiffvg
from config import args
from pathlib import Path
from src import utils
from drawing_model import DrawingModel
import src.fid_score as fid


device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

NUM_TRIALS = 10
args.num_iter = 1000

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

for n, name in enumerate(names):

    args.svg_path = f"data/drawing_{name}.svg"
    args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': yy0[n], 'y1': yy1[n]}

    for trial in range(NUM_TRIALS):

        save_path = Path("results/").joinpath(f'change_prompt/{name}/pA/')
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = Path("results/").joinpath(f'change_prompt/{name}/pAB/')
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = Path("results/").joinpath(f'change_prompt/{name}/pB/')
        save_path.mkdir(parents=True, exist_ok=True)

        args.clip_prompt = prompts_A[n]

        drawing_model = DrawingModel(args, device)
        drawing_model.process_text(args)
        drawing_model.load_svg_shapes(args.svg_path)
        drawing_model.add_random_shapes(args.num_paths, args)
        drawing_model.initialize_variables()
        drawing_model.initialize_optimizer()

        for t in range(args.num_iter):

            drawing_model.run_epoch(t, args)
            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model.losses['global'].item()
            )

        with torch.no_grad():
            pydiffvg.imwrite(
                drawing_model.img,
                f'results/change_prompt/{name}/pA/{trial}.png',
                gamma=1,
            )

        args.clip_prompt = prompts_B[n]
        drawing_model.process_text(args)
        drawing_model.initialize_optimizer()

        for t in range(args.num_iter):
            drawing_model.run_epoch(t, args)
            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model.losses['global'].item()
            )

        with torch.no_grad():
            pydiffvg.imwrite(
                drawing_model.img,
                f'results/change_prompt/{name}/pAB/{trial}.png',
                gamma=1,
            )

        drawing_model = DrawingModel(args, device)
        drawing_model.process_text(args)
        drawing_model.load_svg_shapes(args.svg_path)
        drawing_model.add_random_shapes(args.num_paths, args)
        drawing_model.initialize_variables()
        drawing_model.initialize_optimizer()

        for t in range(args.num_iter):
            drawing_model.run_epoch(t, args)
            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model.losses['global'].item()
            )

        with torch.no_grad():
            pydiffvg.imwrite(
                drawing_model.img,
                f'results/change_prompt/{name}/pB/{trial}.png',
                gamma=1,
            )

import pandas as pd

fid_data = []
for n, name in enumerate(names):
    pA = f'results/change_prompt/{name}/pA'
    pAB = f'results/change_prompt/{name}/pAB'
    pB = f'results/change_prompt/{name}/pB'
    fid_data.append(
        {
            'FID': fid.main([pA, pAB]),
            'name': name,
            'distance': 'to starters',
        }
    )
    fid_data.append(
        {
            'FID': fid.main([pA, pB]),
            'name': name,
            'distance': 'to standard',
        }
    )

df = pd.DataFrame(fid_data)

print(df)

import plotly.express as px

fig = px.scatter(
    df, x="name", y="FID", color="distance", size=[10 for x in range(len(df))]
)
fig.show()
