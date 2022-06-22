import torch
import pydiffvg
from src.config import args
from pathlib import Path
from src import utils
from src.drawing_model import DrawingModel

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

prompt_A = 'A tall red chair.'
prompt_B = 'A short blue chair.'
NUM_ITER = 500
SVG_PATH = "data/drawing_chair.svg"
SAVE_PATH = "test/prompt_changes"
args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': 0.5, 'y1': 1.0}

save_path = Path("results/").joinpath(f'{SAVE_PATH}')
save_path.mkdir(parents=True, exist_ok=True)


class TestPromptChange:
    def test_prompt_change(self):

        # Using prompt A #################
        args.clip_prompt = prompt_A
        drawing_model = DrawingModel(args, device)
        drawing_model.process_text(args)
        drawing_model.load_svg_shapes(args.svg_path)
        drawing_model.add_random_shapes(args.num_paths)
        drawing_model.initialize_variables()
        drawing_model.initialize_optimizer()

        for t in range(NUM_ITER):

            drawing_model.run_epoch(t, args)
            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model.losses['global'].item()
            )

        with torch.no_grad():
            pydiffvg.imwrite(
                drawing_model.img,
                f'results/{SAVE_PATH}/before.png',
                gamma=1,
            )

        # Using prompt B #################
        args.clip_prompt = prompt_B
        drawing_model.process_text(args)

        for t in range(NUM_ITER):
            drawing_model.run_epoch(t, args)
            utils.printProgressBar(
                t + 1, args.num_iter, drawing_model.losses['global'].item()
            )

        with torch.no_grad():
            pydiffvg.imwrite(
                drawing_model.img,
                f'results/{SAVE_PATH}/after.png',
                gamma=1,
            )

        assert True
