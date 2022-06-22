import torch
import copy
from src.config import args
from src.drawing_model import DrawingModel

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

prompt_A = 'A tall red chair.'
prompt_B = 'A short blue chair.'
NUM_ITER = 5
SVG_PATH = "data/drawing_chair.svg"
args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': 0.5, 'y1': 1.0}


class TestPromptChange:
    def test_prompt_change(self):

        # Using prompt A #################
        args.clip_prompt = prompt_A
        drawing_model = DrawingModel(args, device)
        drawing_model.process_text(args)
        text_features_A = copy.copy(drawing_model.text_features)
        drawing_model.load_svg_shapes(args.svg_path)
        drawing_model.add_random_shapes(args.num_paths)
        drawing_model.initialize_variables()
        drawing_model.initialize_optimizer()

        for t in range(NUM_ITER):
            drawing_model.run_epoch(t, args)

        assert torch.norm(drawing_model.text_features - text_features_A) == 0

        # Using prompt B #################
        args.clip_prompt = prompt_B
        drawing_model.process_text(args)

        for t in range(NUM_ITER):
            drawing_model.run_epoch(t, args)

        assert torch.norm(drawing_model.text_features - text_features_A) > 0
