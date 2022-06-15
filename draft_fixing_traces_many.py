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
import copy
import src.experiment_utils as eu

NUM_TRIALS = 30
GENS_PER_TRIAL = 20
NUM_SETS = 5
args.num_iter = 1000
args.w_geo = 10
SAVE_PATH = 'fix_paths4'
names = ['chair', 'hat', 'lamp', 'pot', 'boat', 'dress', 'shoe', 'bust']
yy0 = [0.5, 0.6, 0.0, 0.0, 0.35, 0.0, 0.0, 0.5]
yy1 = [1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0]
prompts = [
    'A red chair.',
    'A drawing of a hat.',
    'A drawing of a lamp.',
    'A drawing of a pot.',
    'A drawing of a boat.',
    'A blue dress.',
    'A high-heel shoe.',
    'A bust.',
]

names = names[4:]
yy0 = yy0[4:]
yy1 = yy1[4:]
prompts = prompts[4:]

for n, name in enumerate(names):

    args.svg_path = f"data/drawing_{name}.svg"
    args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': yy0[n], 'y1': yy1[n]}
    args.clip_prompt = f"A drawing of a {name}"

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    save_path = Path("results/").joinpath(f'{SAVE_PATH}/{name}')
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = str(save_path) + '/'

    for trial in range(NUM_TRIALS):

        drawing_model = DrawingModel(args, device)
        drawing_model.process_text(args)
        drawing_model.load_svg_shapes(args)
        drawing_model.add_random_shapes(args.num_paths, args)
        drawing_model.initialize_variables(args)
        drawing_model.initialize_optimizer()

        t0 = 0
        for t in range(args.num_iter):

            if t + 1 == args.num_iter:
                with torch.no_grad():
                    pydiffvg.imwrite(
                        drawing_model.img,
                        save_path + f'from_scratch/{trial}.png',
                        gamma=1,
                    )

            drawing_model.run_epoch(t, args)

            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model.losses['global'].item()
            )

        if trial < NUM_SETS:
            shapes, shape_groups, fixed_inds = eu.get_fixed_paths(
                drawing_model, args, 6
            )
            for gen in range(GENS_PER_TRIAL):
                drawing_modelC = DrawingModel(args, device)
                drawing_modelC.process_text(args)
                drawing_modelC.load_listed_shapes(
                    args, shapes, shape_groups, fixed_inds
                )
                drawing_modelC.add_random_shapes(args.num_paths, args)
                drawing_modelC.initialize_variables(args)
                new_mask = 1 - torch.floor(drawing_modelC.img0)
                drawing_modelC.mask = torch.round(
                    (drawing_model.mask + new_mask) / 2 + 0.1
                )
                drawing_modelC.initialize_optimizer()
                with torch.no_grad():
                    pydiffvg.imwrite(
                        drawing_modelC.img0.squeeze(0).permute(1, 2, 0).cpu(),
                        save_path + f'fixe_lines/{trial}.png',
                        gamma=1,
                    )
                    pydiffvg.imwrite(
                        drawing_modelC.mask.detach().squeeze(0).permute(1, 2, 0).cpu(),
                        save_path + f'fixe_lines/{trial}.png',
                        gamma=1,
                    )
                for t in range(args.num_iter):

                    drawing_modelC.run_epoch(t, args)

                    if t + 1 == args.num_iter:
                        with torch.no_grad():
                            pydiffvg.imwrite(
                                drawing_modelC.img,
                                save_path + f'gen_trial{trial}/{gen}.png',
                                gamma=1,
                            )

                    utils.printProgressBar(
                        t + 1, args.num_iter, drawing_modelC.losses['global'].item()
                    )
