import torch
import copy
from src.config import args
from src.drawing_model import Cicada

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

NUM_ITER = 5


class TestTraceFixing:
    def test_prompt_change(self):

        cicada = Cicada(args, device)
        cicada.process_text(args)
        cicada.load_svg_shapes(args.svg_path)
        cicada.add_random_shapes(args.num_paths)
        cicada.initialize_variables()
        cicada.initialize_optimizer()

        for t in range(NUM_ITER):
            cicada.run_epoch(t, args)

        with torch.no_grad():
            traces = copy.deepcopy(cicada.drawing.traces[1:3])

        for t in range(NUM_ITER):
            cicada.run_epoch(t, args)

        with torch.no_grad():
            for trace in traces:
                points = cicada.drawing.traces[
                    trace.shape_group.shape_ids.item()
                ].shape.points.detach()
                assert torch.norm(trace.shape.points - points) > 0

        cicada.add_traces(traces, replace=True)

        with torch.no_grad():
            for trace in traces:
                points = cicada.drawing.traces[
                    trace.shape_group.shape_ids.item()
                ].shape.points.detach()
                assert torch.norm(trace.shape.points - points) == 0

        N = len(cicada.drawing.traces)
        cicada.add_traces(traces)
        assert (
            cicada.drawing.traces[-1].shape_group.shape_ids.item() == N + 1
        )  # cause I added 2 traces

    def test_trace_removal(self):
        cicada = Cicada(args, device)
        cicada.process_text(args)
        cicada.load_svg_shapes(args.svg_path)
        cicada.add_random_shapes(args.num_paths)
        cicada.initialize_variables()
        cicada.initialize_optimizer()

        for t in range(NUM_ITER):
            cicada.run_epoch(t, args)

        N = cicada.drawing.traces[-1].shape_group.shape_ids.item()
        cicada.remove_traces([2, 3])
        assert cicada.drawing.traces[-1].shape_group.shape_ids.item() == N - 2
