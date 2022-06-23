import torch
import copy
import pydiffvg
from src.config import args
from src.drawing_model import DrawingModel
from src import experiment_utils as eu

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

NUM_ITER = 500
SAVE_PATH = "test/trace_fixing"
N_KEEP = 5

drawing_model = DrawingModel(args, device)
drawing_model.process_text(args)
drawing_model.load_svg_shapes(args.svg_path)
drawing_model.add_random_shapes(args.num_paths)
drawing_model.initialize_variables()
drawing_model.initialize_optimizer()

for t in range(NUM_ITER):
    drawing_model.run_epoch(t, args)

with torch.no_grad():
    pydiffvg.imwrite(
        drawing_model.img,
        f'results/{SAVE_PATH}/before.png',
        gamma=1,
    )

inds = list(range(-5, -1))


with torch.no_grad():
    for i in inds:
        drawing_model.drawing.traces[i].is_fixed = True
        drawing_model.points_vars[i].requires_grad = False
        drawing_model.stroke_width_vars[i].requires_grad = False
        drawing_model.color_vars[i].requires_grad = False

for t in range(NUM_ITER):
    drawing_model.run_epoch(t, args)
    
with torch.no_grad():
    pydiffvg.imwrite(
        drawing_model.img,
        f'results/{SAVE_PATH}/after.png',
        gamma=1,
    )