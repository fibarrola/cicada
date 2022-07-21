import torch
import pydiffvg
import os
import numpy as np
import src.fid_score as fid
import pickle
import datetime
from pathlib import Path
from src import utils
from src.config import args
from src.drawing_model import Cicada
from src.style import image_loader


NUM_TRIALS = 20
SAVE_PATH = 'penalization_effect'
CREATE_SAMPLES = True
args.num_iter = 1500

if CREATE_SAMPLES:
    SAVE_PATH += (datetime.datetime.today() + datetime.timedelta(hours=11)).strftime(
            "%Y_%m_%d_%H_%M"
        )
else:
    SAVE_PATH += '2022_07_02_02_16'


names = ['chair', 'hat', 'lamp', 'pot', 'boat', 'dress', 'shoe', 'bust']
yy0 = [0.5, 0.6, 0.0, 0.0, 0.35, 0.0, 0.0, 0.5]
yy1 = [1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.55, 1.0]
prompts = [
    'A red chair.',
    'A drawing of a hat.',
    'A drawing of a lamp.',
    'A drawing of a pot.',
    'A drawing of a boat.',
    'A blue dress.',
    'A high-heel shoe.',
    'A bust.',
]

# args.w_points = 0.5 * args.w_points
# args.w_colors = 0.5 * args.w_colors
# args.w_widths = 0.5 * args.w_widths
ww_geo = [0.035, 3.5, 3500]

# names = names[:3]

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

if CREATE_SAMPLES:
    for n, name in enumerate(names):
        args.svg_path = f"data/drawing_{name}.svg"
        args.drawing_area = {'x0': args.x0, 'x1': args.x1, 'y0': yy0[n], 'y1': yy1[n]}
        args.prompt = prompts[n]

        for w, w_geo in enumerate(ww_geo):

            args.w_geo = w_geo

            save_path = Path("results/").joinpath(f'{SAVE_PATH}/{name}/{w}')
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = str(save_path) + '/'

            save_path_prepruning = Path("results/").joinpath(f'{SAVE_PATH}/{name}/pp{w}')
            save_path_prepruning.mkdir(parents=True, exist_ok=True)
            save_path_prepruning = str(save_path_prepruning) + '/'

            for trial in range(NUM_TRIALS):

                cicada = Cicada(args, device)
                cicada.process_text(args)
                cicada.load_svg_shapes(args.svg_path)
                cicada.add_random_shapes(args.num_paths)
                cicada.initialize_variables()
                cicada.initialize_optimizer()

                for t in range(args.num_iter):

                    cicada.run_epoch(t, args)

                    # if t % 10 == 0:
                    utils.printProgressBar(
                        t + 1, args.num_iter, cicada.losses['global'].item()
                    )

                    if t == round(args.num_iter*0.75):
                        cicada.prune(0.5)
                        with torch.no_grad():
                            pydiffvg.imwrite(
                                cicada.img,
                                save_path_prepruning + f'prep/{trial}.png',
                                gamma=1,
                            )

                with torch.no_grad():
                    pydiffvg.imwrite(
                        cicada.img,
                        save_path + f'{trial}.png',
                        gamma=1,
                    )


cov_data = []
loss_data = []
for n, name in enumerate(names):

    args.prompt = prompts[n]
    drawing_model = Cicada(args, device)
    drawing_model.process_text(args)

    for w, w_geo in enumerate(ww_geo):
        for rand_iter in range(3):
            mu, S = fid.get_statistics(
                f"results/{SAVE_PATH}/{name}/{w}",  # rand_sampled_set_dim=5
            )
            C = 0.5 * S.shape[0] * (np.log(2 * np.pi) + 1)
            aux, _ = np.linalg.eigh(S)
            print(len([a for a in aux if a > 1e-6]))
            aux = [np.log(a) for a in aux if a > 1e-6]
            aux = 0.5 * np.nansum(np.log(aux))
            cov_data.append(
                {
                    'Entropy': C + aux,
                    'name': name,
                    'penalizer': w_geo,
                }
            )

        filenames = os.listdir(f"results/{SAVE_PATH}/{name}/{w}")
        for filename in filenames:
            img = image_loader(f"results/{SAVE_PATH}/{name}/{w}/{filename}")
            img_augs = []
            for n in range(args.num_augs):
                img_augs.append(drawing_model.augment_trans(img))
            im_batch = torch.cat(img_augs)
            img_features = drawing_model.model.encode_image(im_batch)
            loss = 0
            for n in range(args.num_augs):
                loss -= torch.cosine_similarity(
                    drawing_model.text_features, img_features[n : n + 1], dim=1
                )
            loss_data.append(
                {
                    'loss': loss.detach().cpu().item(),
                    'name': name,
                    'penalizer': w_geo,
                }
            )


with open(f'results/{SAVE_PATH}/cov_data.pkl', 'wb') as f:
    pickle.dump(cov_data, f)
with open(f'results/{SAVE_PATH}/loss_data.pkl', 'wb') as f:
    pickle.dump(loss_data, f)

# fig = go.Figure()

# colors = [
#     'rgba(0,170,123,1)',
#     'rgba(0,83,170,1)',
#     'rgba(255,157,0,1)',
#     'rgba(255,157,0,1)',
#     'rgba(255,157,0,1)',
# ]
# for w, w_geo in enumerate(ww_geo):
#     xx = [x['name'] for x in cov_data if x['penalizer'] == w_geo]
#     yy = [x['Entropy'] for x in cov_data if x['penalizer'] == w_geo]
#     fig.add_trace(go.Box(y=yy, x=xx, name=f'geo pen = {w_geo}', marker_color=colors[w]))

# fig.update_layout(
#     boxmode='group',
#     legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
# )
# fig.show()

# fig = go.Figure()
# names = ['Low penalization', 'Medium penalization', 'High penalization']
# colors = ['rgba(0,170,123,1)', 'rgba(0,83,170,1)', 'rgba(255,157,0,1)', 'rgba(255,157,0,1)', 'rgba(255,157,0,1)']
# for w, w_geo in enumerate(ww_geo):
#     xx = ['nothing' for x in cov_data if x['penalizer']==w_geo]
#     yy = [x['Entropy'] for x in cov_data if x['penalizer']==w_geo]
#     fig.add_trace(go.Box(y=yy, x=xx, name=names[w], marker_color=colors[w]))
# fig.update_xaxes(visible=False)
# fig.update_yaxes(title_text='Norm of Cov. Matrix')
# fig.update_layout(
#     boxmode='group',
#     legend={'yanchor': "top", 'y': 0.99, 'xanchor': "right", 'x': 0.99},
# )
# fig.show()
