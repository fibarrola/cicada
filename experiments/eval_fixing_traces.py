import pandas as pd
import torch
import src.fid_score as fid
import plotly.graph_objects as go
import os
import pickle
from src import utils
from src.style import image_loader
from src.config import args
from src.drawing_model import Cicada


COMPUTE = False
PATH = 'fix_paths4'

names = ['chair', 'hat', 'lamp', 'pot', 'boat', 'dress', 'shoe', 'bust']
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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if not COMPUTE:
    with open(f'results/{PATH}/entropy.pkl', 'rb') as f:
        cov_data = pickle.load(f)
    with open(f'results/{PATH}/loss_data.pkl', 'rb') as f:
        loss_data = pickle.load(f)

else:
    cov_data = []
    loss_data = []
    for n, name in enumerate(names):

        args.clip_prompt = prompts[n]
        drawing_model = Cicada(args, device)
        drawing_model.process_text(args)

        for process_name in [
            'from_scratch',
            'from_scratch',
            'from_scratch',
            'from_scratch',
            'from_scratch',
            'gen_trial0',
            'gen_trial1',
            'gen_trial2',
            'gen_trial3',
            'gen_trial4',
        ]:
            subset_dim = 20 if process_name == 'from_scratch' else None
            mu, S = fid.get_statistics(
                f"results/fix_paths4/{name}/{process_name}",
                rand_sampled_set_dim=subset_dim,
            )
            gen_type = (
                'standard' if process_name == 'from_scratch' else 'trace-conditioned'
            )
            cov_data.append(
                {
                    'Entropy': utils.tie(S),
                    'name': name,
                    'generation': gen_type,
                }
            )

            filenames = os.listdir(f"results/fix_paths4/{name}/{process_name}")
            for filename in filenames:
                img = image_loader(
                    f"results/fix_paths4/{name}/{process_name}/{filename}"
                )
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
                        'generation': gen_type,
                    }
                )

    with open(f'results/{PATH}/entropy.pkl', 'wb') as f:
        pickle.dump(cov_data, f)
    with open(f'results/{PATH}/loss_data.pkl', 'wb') as f:
        pickle.dump(loss_data, f)


# FOR ENTROPY
fig = go.Figure()
xx = [x['name'] for x in cov_data if x['generation'] == 'standard']
yy = [x['Entropy'] for x in cov_data if x['generation'] == 'standard']
fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color=utils.color_palette[0]))

xx = [x['name'] for x in cov_data if x['generation'] == 'trace-conditioned']
yy = [x['Entropy'] for x in cov_data if x['generation'] == 'trace-conditioned']
fig.add_trace(
    go.Box(y=yy, x=xx, name='trace-conditioned', marker_color=utils.color_palette[2])
)
fig.update_layout(
    boxmode='group',
    yaxis_title="TIE",
    legend={'yanchor': "bottom", 'y': 0.01, 'xanchor': "right", 'x': 0.99},
    font={'size': 16},
    margin={'b': 2, 'l': 5, 'r': 2, 't': 2},
)
fig.show()


# FOR LOSSES
df = pd.DataFrame(loss_data)
fig = go.Figure()

xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['loss']

fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color=utils.color_palette[0]))

xx = df.query('generation=="trace-conditioned"')['name']
yy = df.query('generation=="trace-conditioned"')['loss']

fig.add_trace(
    go.Box(y=yy, x=xx, name='trace-conditioned', marker_color=utils.color_palette[2])
)

fig.update_layout(
    boxmode='group',
    yaxis_title="Semantic Loss",
    legend={'yanchor': "bottom", 'y': 0.01, 'xanchor': "right", 'x': 0.99},
    font={'size': 16},
    margin={'b': 2, 'l': 2, 'r': 2, 't': 2},
)
fig.show()
