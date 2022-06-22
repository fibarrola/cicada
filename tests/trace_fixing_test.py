import torch
import copy
from src.config import args
from src.drawing_model import DrawingModel

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

NUM_ITER = 5


class TestTraceFixing:
    def test_prompt_change(self):

        drawing_model = DrawingModel(args, device)
        drawing_model.process_text(args)
        drawing_model.load_svg_shapes(args.svg_path)
        drawing_model.add_random_shapes(args.num_paths)
        drawing_model.initialize_variables()
        drawing_model.initialize_optimizer()

        for t in range(NUM_ITER):
            drawing_model.run_epoch(t, args)

        with torch.no_grad():
            traces = copy.deepcopy(drawing_model.drawing.traces[1:3])

        for t in range(NUM_ITER):
            drawing_model.run_epoch(t, args)

        with torch.no_grad():
            for trace in traces:
                points = drawing_model.drawing.traces[
                    trace.shape_group.shape_ids.item()
                ].shape.points.detach()
                assert torch.norm(trace.shape.points - points) > 0

        drawing_model.add_traces(traces, replace=True)

        with torch.no_grad():
            for trace in traces:
                points = drawing_model.drawing.traces[
                    trace.shape_group.shape_ids.item()
                ].shape.points.detach()
                assert torch.norm(trace.shape.points - points) == 0

        N = len(drawing_model.drawing.traces)
        drawing_model.add_traces(traces)
        assert (
            drawing_model.drawing.traces[-1].shape_group.shape_ids.item() == N + 1
        )  # cause I added 2 traces
