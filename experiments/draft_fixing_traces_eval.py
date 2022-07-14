import numpy as np
import pandas as pd
import torch
import src.fid_score as fid
import plotly.express as px
import plotly.graph_objects as go
import os
from src.style import image_loader
from src.config import args
from src.drawing_model import Cicada


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

cov_data = []
loss_data = []
# names = names[:1]
# for n, name in enumerate(names):

#     args.clip_prompt = prompts[n]
#     drawing_model = Cicada(args, device)
#     drawing_model.process_text(args)

#     for process_name in [
#         'from_scratch',
#         'from_scratch',
#         'from_scratch',
#         'from_scratch',
#         'from_scratch',
#         'gen_trial0',
#         'gen_trial1',
#         'gen_trial2',
#         'gen_trial3',
#         'gen_trial4',
#     ]:
#         subset_dim = 20 if process_name == 'from_scratch' else None
#         mu, S = fid.get_statistics(
#             f"results/fix_paths4/{name}/{process_name}", rand_sampled_set_dim=subset_dim
#         )
#         gen_type = 'standard' if process_name == 'from_scratch' else 'trace-conditioned'
#         C = 0.5*S.shape[0]*(np.log(2*np.pi)+ 1)
#         aux, _ = np.linalg.eigh(S)
#         print(len([a for a in aux if a>1e-6]))
#         aux = [np.log(a) for a in aux if a>1e-6 ]
#         aux = np.nansum(np.log(aux))

#         cov_data.append(
#             {
#                 'Entropy': C + aux,
#                 'name': name,
#                 'generation': gen_type,
#             }
#         )

#         filenames = os.listdir(f"results/fix_paths4/{name}/{process_name}")
#         for filename in filenames:
#             img = image_loader(f"results/fix_paths4/{name}/{process_name}/{filename}")
#             img_augs = []
#             for n in range(args.num_augs):
#                 img_augs.append(drawing_model.augment_trans(img))
#             im_batch = torch.cat(img_augs)
#             img_features = drawing_model.model.encode_image(im_batch)
#             loss = 0
#             for n in range(args.num_augs):
#                 loss -= torch.cosine_similarity(
#                     drawing_model.text_features, img_features[n : n + 1], dim=1
#                 )
#             loss_data.append(
#                 {
#                     'loss': loss.detach().cpu().item(),
#                     'name': name,
#                     'generation': gen_type,
#                 }
#             )


import pickle

# with open('results/fix_paths4/cov_data.pkl', 'wb') as f:
#     pickle.dump(cov_data, f)
# with open('results/fix_paths4/loss_data.pkl', 'wb') as f:
#     pickle.dump(loss_data, f)

with open('results/fix_paths4/entropy.pkl','rb') as f:
    cov_data = pickle.load(f)

df = pd.DataFrame(cov_data)

# fig = px.scatter(
#     df,
#     x="name",
#     y="Covariance Norm",
#     color="generation",  # size=[2 for x in range(len(df))]
# )
# fig.show()

# fig = px.box(
#     df,
#     x="name",
#     y="Covariance Norm",
#     color="generation",  # size=[2 for x in range(len(df))]
# )
# fig.show()

fig = go.Figure()


xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['Entropy']
fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color='rgba(255,157,0,1)'))

xx = df.query('generation=="trace-conditioned"')['name']
yy = df.query('generation=="trace-conditioned"')['Entropy']
fig.add_trace(
    go.Box(y=yy, x=xx, name='trace-conditioned', marker_color='rgba(0,83,170,1)')
)

fig.update_layout(
    boxmode='group',
    yaxis_title="TIE",
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
    font={'size': 16},
)
fig.show()


# NOw FOR LOSSES
df = pd.DataFrame(loss_data)
fig = go.Figure()

xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['loss']

fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color='rgba(255,157,0,1)'))

xx = df.query('generation=="trace-conditioned"')['name']
yy = df.query('generation=="trace-conditioned"')['loss']

fig.add_trace(
    go.Box(y=yy, x=xx, name='trace-conditioned', marker_color='rgba(0,83,170,1)')
)

fig.update_layout(
    boxmode='group',
    yaxis_title="Semantic Loss",
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
    font={'size': 16},
)
fig.show()
