import torch
from src.config import args
from src import utils
from src.drawing_model import Cicada
import src.fid_score as fid
import pandas as pd
import pickle
import plotly.graph_objects as go
import os
from src.style import image_loader

COMPUTE = False
PATH = 'prompt_change'

device = "cuda:0" if torch.cuda.is_available() else "cpu"


if not COMPUTE:
    with open(f'results/{PATH}/entropy.pkl', 'rb') as f:
        cov_data = pickle.load(f)
    with open(f'results/{PATH}/loss_data.pkl', 'rb') as f:
        loss_data = pickle.load(f)

else:
    NUM_SETS = 5
    names = ['chair', 'hat', 'lamp', 'pot', 'boat', 'dress', 'shoe', 'bust']
    prompts_B = [
        'A short blue chair.',
        'A drawing of a flat pink hat.',
        'A drawing of a round black lamp.',
        'A drawing of a large pot.',
        'A drawing of a steam boat.',
        'A blue and green long dress.',
        'A high-heel black shoe.',
        'A bust of an army general.',
    ]
    cov_data = []
    loss_data = []
    for n, name in enumerate(names):

        args.clip_prompt = prompts_B[n]
        drawing_model = Cicada(args, device)
        drawing_model.process_text(args)

        for process_name in ['pB' for ts in range(NUM_SETS)] + [
            f'pAB_{ts}' for ts in range(NUM_SETS)
        ]:
            subset_dim = 10 if process_name == 'pB' else None
            mu, S = fid.get_statistics(
                f"results/prompt_change/{name}/{process_name}",
                rand_sampled_set_dim=subset_dim,
            )
            gen_type = (
                'standard' if process_name == 'pB' else 'prompt-change-conditioned'
            )
            cov_data.append(
                {
                    'Entropy': utils.tie(S),
                    'name': name,
                    'generation': gen_type,
                }
            )

            filenames = os.listdir(f"results/prompt_change/{name}/{process_name}")
            for filename in filenames:
                img = image_loader(
                    f"results/prompt_change/{name}/{process_name}/{filename}"
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


df = pd.DataFrame(cov_data)

fig = go.Figure()
xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['Entropy']
fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color=utils.color_palette[0]))

xx = df.query('generation=="prompt-change-conditioned"')['name']
yy = df.query('generation=="prompt-change-conditioned"')['Entropy']
fig.add_trace(
    go.Box(
        y=yy,
        x=xx,
        name='prompt-change-conditioned',
        marker_color=utils.color_palette[1],
    )
)

fig.update_layout(
    boxmode='group',
    yaxis_title="TIE",
    legend={'yanchor': "bottom", 'y': 0.01, 'xanchor': "left", 'x': 0.01},
    font={'size': 16},
    margin={'b': 2, 'l': 2, 'r': 2, 't': 2},
)
fig.show()

# NOw FOR LOSSES
df = pd.DataFrame(loss_data)
fig = go.Figure()

xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['loss']

fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color=utils.color_palette[0]))

xx = df.query('generation=="prompt-change-conditioned"')['name']
yy = df.query('generation=="prompt-change-conditioned"')['loss']

fig.add_trace(
    go.Box(
        y=yy,
        x=xx,
        name='prompt-change-conditioned',
        marker_color=utils.color_palette[1],
    )
)

fig.update_layout(
    boxmode='group',
    yaxis_title="Semantic Loss",
    legend={'yanchor': "bottom", 'y': 0.01, 'xanchor': "right", 'x': 0.99},
    font={'size': 16},
    margin={'b': 2, 'l': 2, 'r': 2, 't': 2},
)
fig.show()

# df = pd.DataFrame(loss_data)
# fig = go.Figure()

# xx = df.query('generation=="standard"')['name']
# yy = df.query('generation=="standard"')['loss']

# fig.add_trace(
#     go.Box(
#         y=yy,
#         x=xx,
#         name='standard',
#         marker_color='rgba(255,157,0,1)',
#         quartilemethod="inclusive",
#     )
# )

# xx = df.query('generation=="prompt-change-conditioned"')['name']
# yy = df.query('generation=="prompt-change-conditioned"')['loss']

# fig.add_trace(
#     go.Box(
#         y=yy,
#         x=xx,
#         name='prompt-change-conditioned',
#         marker_color='rgba(0,83,170,1)',
#         quartilemethod="inclusive",
#     )
# )

# fig.update_layout(
#     boxmode='group',
#     yaxis_title="Semantic Loss",
#     legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
#     font={'size': 16},
# )
# fig.show()

# df = pd.DataFrame(loss_data)
# fig = go.Figure()

# xx = df.query('generation=="standard"')['name']
# yy = df.query('generation=="standard"')['loss']

# fig.add_trace(
#     go.Box(
#         y=yy,
#         x=xx,
#         name='standard',
#         marker_color='rgba(255,157,0,1)',
#         quartilemethod="exclusive",
#     )
# )

# xx = df.query('generation=="prompt-change-conditioned"')['name']
# yy = df.query('generation=="prompt-change-conditioned"')['loss']

# fig.add_trace(
#     go.Box(
#         y=yy,
#         x=xx,
#         name='prompt-change-conditioned',
#         marker_color='rgba(0,83,170,1)',
#         quartilemethod="exclusive",
#     )
# )

# fig.update_layout(
#     boxmode='group',
#     yaxis_title="Semantic Loss",
#     legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
#     font={'size': 16},
# )
# fig.show()
