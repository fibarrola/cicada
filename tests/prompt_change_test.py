import torch
import copy
from src.config import args
from src.drawing_model import Cicada

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

prompt_A = 'A tall red chair.'
prompt_B = 'A short blue chair.'
NUM_ITER = 5
SVG_PATH = "data/drawing_chair.svg"
args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': 0.5, 'y1': 1.0}


class TestPromptChange:
    def test_prompt_change(self):

        # Using prompt A #################
        args.prompt = prompt_A
        cicada = Cicada(args, device)
        cicada.process_text(args)
        text_features_A = copy.copy(cicada.text_features)
        cicada.load_svg_shapes(args.svg_path)
        cicada.add_random_shapes(args.num_paths)
        cicada.initialize_variables()
        cicada.initialize_optimizer()

        for t in range(NUM_ITER):
            cicada.run_epoch(t, args)

        assert torch.norm(cicada.text_features - text_features_A) == 0

        # Using prompt B #################
        args.prompt = prompt_B
        cicada.process_text(args)

        for t in range(NUM_ITER):
            cicada.run_epoch(t, args)

        assert torch.norm(cicada.text_features - text_features_A) > 0
