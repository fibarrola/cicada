import torch
import pydiffvg
from config import args
from pathlib import Path
from src import utils
from drawing_model import DrawingModel

NUM_TRIALS = 10
# args.num_iter = 100

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

save_path = Path("results/").joinpath('fix_paths')
save_path.mkdir(parents=True, exist_ok=True)
save_path = str(save_path) + '/'

for trial in range(NUM_TRIALS):

    drawing_model = DrawingModel(args, device)
    drawing_model.process_text(args)
    drawing_model.initialize_shapes(args)
    drawing_model.initialize_variables(args)
    drawing_model.initialize_optimizer()

    t0 = 0
    for t in range(args.num_iter):

        if (t + 1) % args.num_iter // 20:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + f'_00_{trial}.png', gamma=1,
                )

        if t == args.num_iter // 2:
            shapes, shape_groups, fixed_inds = drawing_model.get_fixed_paths(args, 3)
            # del drawing_model
            # drawing_model = DrawingModel(args, device)
            # drawing_model.process_text(args)
            drawing_model.load_shapes(args, shapes, shape_groups, fixed_inds)
            drawing_model.initialize_variables(args)
            drawing_model.initialize_optimizer()
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img0.squeeze(0).permute(1, 2, 0).cpu(),
                    save_path + f'_fixes_{trial}.png',
                    gamma=1,
                )

        drawing_model.run_epoch(t, args)

        utils.printProgressBar(
            t + 1, args.num_iter, drawing_model.losses['global'].item()
        )

        if t == args.num_iter // 2 + 1:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + f'_fix_init_{trial}.png', gamma=1,
                )

    continue
    drawing_model.process_text(args)
    drawing_model.initialize_optimizer()

    for t in range(args.num_iter):

        if t == 1:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + f'start_01_{trial}_00.png', gamma=1,
                )

        if (t + 1) % args.num_iter // 20:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + f'_01_{trial}.png', gamma=1,
                )

        drawing_model.run_epoch(t, args)

        utils.printProgressBar(
            t + 1, args.num_iter, drawing_model.losses['global'].item()
        )

        if t == args.num_iter // 2:
            drawing_model.fix_paths(args, 3)

    drawing_model = DrawingModel(args, device)
    drawing_model.process_text(args)
    drawing_model.initialize_shapes(args)
    drawing_model.initialize_variables(args)
    drawing_model.initialize_optimizer()

    for t in range(args.num_iter):

        if t == 1:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + f'start_11_{trial}_00.png', gamma=1,
                )

        if (t + 1) % args.num_iter // 20:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + f'_11_{trial}.png', gamma=1,
                )

        drawing_model.run_epoch(t, args)

        utils.printProgressBar(
            t + 1, args.num_iter, drawing_model.losses['global'].item()
        )
