import torch
import pydiffvg
from src.config import args
from src.drawing_model import Cicada

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

NUM_ITER = 500
SAVE_PATH = "test/trace_fixing"
N_KEEP = 5

cicada = Cicada(args, device)
cicada.process_text(args)
cicada.load_svg_shapes(args.svg_path)
cicada.add_random_shapes(args.num_paths)
cicada.initialize_variables()
cicada.initialize_optimizer()

for t in range(NUM_ITER):
    cicada.run_epoch(t, args)

with torch.no_grad():
    pydiffvg.imwrite(
        cicada.img,
        f'results/{SAVE_PATH}/before.png',
        gamma=1,
    )

inds = list(range(-5, -1))


with torch.no_grad():
    for i in inds:
        cicada.drawing.traces[i].is_fixed = True
        cicada.points_vars[i].requires_grad = False
        cicada.stroke_width_vars[i].requires_grad = False
        cicada.color_vars[i].requires_grad = False

for t in range(NUM_ITER):
    cicada.run_epoch(t, args)

with torch.no_grad():
    pydiffvg.imwrite(
        cicada.img,
        f'results/{SAVE_PATH}/after.png',
        gamma=1,
    )
