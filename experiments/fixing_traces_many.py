import torch
import pydiffvg
from src import utils
from src.config import args
from pathlib import Path
from src.drawing_model import Cicada
import src.experiment_utils as eu

NUM_TRIALS = 3  # 30
GENS_PER_TRIAL = 2  # 20
NUM_SETS = 1  # 5
args.num_iter = 10  # 00
args.w_geo = 10
SAVE_PATH = 'fix_paths7'
names = ['chair', 'hat', 'lamp', 'pot', 'boat', 'dress', 'shoe', 'bust']
yy0 = [0.5, 0.6, 0.0, 0.0, 0.35, 0.0, 0.0, 0.5]
yy1 = [1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.55, 1.0]
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

        cicada = Cicada(args, device)
        cicada.process_text(args)
        cicada.load_svg_shapes(args.svg_path)
        cicada.add_random_shapes(args.num_paths)
        cicada.initialize_variables()
        cicada.initialize_optimizer()

        t0 = 0
        for t in range(args.num_iter):

            if t + 1 == args.num_iter:
                with torch.no_grad():
                    pydiffvg.imwrite(
                        cicada.img,
                        save_path + f'from_scratch/{trial}.png',
                        gamma=1,
                    )

            cicada.run_epoch(t, args)

            utils.printProgressBar(t + 1, args.num_iter, cicada.losses['global'].item())

        if trial < NUM_SETS:
            shapes, shape_groups = eu.get_fixed_paths(cicada, 6)
            for gen in range(GENS_PER_TRIAL):
                cicadaC = Cicada(args, device)
                cicadaC.process_text(args)
                cicadaC.load_listed_shapes(shapes, shape_groups, fix=True)
                cicadaC.add_random_shapes(args.num_paths)
                cicadaC.initialize_variables()
                new_mask = 1 - torch.floor(cicadaC.img0)
                cicadaC.mask = torch.round((cicada.mask + new_mask) / 2 + 0.1)
                cicadaC.initialize_optimizer()
                with torch.no_grad():
                    pydiffvg.imwrite(
                        cicadaC.img0.squeeze(0).permute(1, 2, 0).cpu(),
                        save_path + f'fixe_lines/{trial}.png',
                        gamma=1,
                    )
                    pydiffvg.imwrite(
                        cicadaC.mask.detach().squeeze(0).permute(1, 2, 0).cpu(),
                        save_path + f'fixe_lines/{trial}.png',
                        gamma=1,
                    )
                for t in range(args.num_iter):

                    cicadaC.run_epoch(t, args)

                    if t + 1 == args.num_iter:
                        with torch.no_grad():
                            pydiffvg.imwrite(
                                cicadaC.img,
                                save_path + f'gen_trial{trial}/{gen}.png',
                                gamma=1,
                            )

                    utils.printProgressBar(
                        t + 1, args.num_iter, cicadaC.losses['global'].item()
                    )
