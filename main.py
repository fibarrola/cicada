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

drawing_model = DrawingModel(args, device)
drawing_model.process_text(args)

t0 = time.time()

for trial in range(args.num_trials):
    time_str = (datetime.datetime.today() + datetime.timedelta(hours=11)).strftime(
        "%Y_%m_%d_%H_%M_%S"
    )

    drawing_model.initialize_shapes(args)
    drawing_model.initialize_variables(args)
    drawing_model.initialize_optimizer()

    # Run the main optimization loop
    for t in range(args.num_iter):

        if (t + 1) % args.num_iter // 20:
            with torch.no_grad():
                pydiffvg.imwrite(
                    drawing_model.img, save_path + time_str + '.png', gamma=1,
                )

        drawing_model.run_epoch(t, args)

        # Pruning
        if args.prune_ratio > 0:
            if t == round(args.num_iter * 0.8):
                with torch.no_grad():
                    pydiffvg.imwrite(
                        drawing_model.img, save_path + time_str + '_preP.png', gamma=1,
                    )
                drawing_model.prune(args)

            if t == round(args.num_iter * 0.8) + 1:
                with torch.no_grad():
                    pydiffvg.imwrite(
                        drawing_model.img, save_path + time_str + '_postP.png', gamma=1,
                    )

        # Print stuff
        # if t % 50 == 0:
        #     print(f"Iteration {t}")
        #     for loss_name in drawing_model.losses:
        #         if loss_name == 'geometric':
        #             for gl_name in drawing_model.losses[loss_name]:
        #                 print(
        #                     f'geo loss {gl_name}: {drawing_model.losses[loss_name][gl_name].item()}'
        #                 )
        #         else:
        #             print(f"{loss_name}: {drawing_model.losses[loss_name].item()}")

        #         im_norm = drawing_model.img_features / drawing_model.img_features.norm(
        #             dim=-1, keepdim=True
        #         )
        #         noun_norm = (
        #             drawing_model.nouns_features
        #             / drawing_model.nouns_features.norm(dim=-1, keepdim=True)
        #         )
        #         similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
        #         values, indices = similarity[0].topk(5)
        #         print("\nTop predictions:\n")
        #         for value, index in zip(values, indices):
        #             print(
        #                 f"{drawing_model.nouns[index]:>16s}: {100 * value.item():.2f}%"
        #             )

        utils.printProgressBar(
            t + 1, args.num_iter, drawing_model.losses['global'].item()
        )

    pydiffvg.imwrite(
        drawing_model.img, save_path + time_str + '.png', gamma=1,
    )
    utils.save_data(save_path, time_str, args)

time_sec = round(time.time() - t0)
print(f"Elapsed time: {time_sec//60} min, {time_sec-60*(time_sec//60)} seconds.")
