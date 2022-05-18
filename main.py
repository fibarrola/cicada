from drawing_model import DrawingModel
import torch
import pydiffvg
import datetime
import time
from src import versions, utils
from config import args
from pathlib import Path

versions.getinfo(showme=False)
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

# Build dir if does not exist & make sure using a
# trailing / or not does not matter
save_path = Path("results/").joinpath(args.dir)
save_path.mkdir(parents=True, exist_ok=True)
save_path = str(save_path) + '/'


t0 = time.time()

prune_places = [
    round(args.num_iter * (k + 1) * 0.8 / args.n_prunes) for k in range(args.n_prunes)
]
p0 = args.prune_ratio


gif_builder = utils.GifBuilder()

for trial in range(args.num_trials):

    args.prune_ratio = p0 / len(prune_places)

    drawing_model = DrawingModel(args, device)
    drawing_model.process_text(args)

    time_str = (datetime.datetime.today() + datetime.timedelta(hours=11)).strftime(
        "%Y_%m_%d_%H_%M_%S"
    )

    drawing_model.initialize_shapes(args)
    drawing_model.initialize_variables(args)
    drawing_model.initialize_optimizer()

    # Run the main optimization loop
    for t in range(args.num_iter):

        if t == 1:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + time_str + '00.png', gamma=1,
                )

        if (t + 1) % args.num_iter // 20:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + time_str + '.png', gamma=1,
                )
                if args.build_gif:
                    gif_builder.add(drawing_model.img)

        drawing_model.run_epoch(t, args)

        # Pruning
        if t in prune_places:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + time_str + f'_preP_{t}.png', gamma=1,
                )
            drawing_model.prune(args)
            args.prune_ratio += p0 / len(prune_places)

        if t - 1 in prune_places:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img,
                    save_path + time_str + f'_postP_{t-1}.png',
                    gamma=1,
                )

        utils.printProgressBar(
            t + 1, args.num_iter, drawing_model.losses['global'].item()
        )

    pydiffvg.imwrite(
        drawing_model.img, save_path + time_str + '.png', gamma=1,
    )
    utils.save_data(save_path, time_str, args)


if args.build_gif:
    gif_builder.build_gif(save_path + time_str)

time_sec = round(time.time() - t0)
print(f"Elapsed time: {time_sec//60} min, {time_sec-60*(time_sec//60)} seconds.")
