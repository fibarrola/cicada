import plotly.graph_objects as go
import pickle
import torch
from src import utils

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

# DATA_PATH = 'penalization_effect2/2022_07_16_05_42'
DATA_PATH = 'penalization_effect2/2022_07_18_22_02'

with open(f'results/{DATA_PATH}/cov_data.pkl', 'rb') as f:
    cov_data = pickle.load(f)
with open(f'results/{DATA_PATH}/loss_data.pkl', 'rb') as f:
    loss_data = pickle.load(f)

ww_geo = list(set([x['penalizer'] for x in cov_data]))

fig = go.Figure()

for w, w_geo in enumerate(ww_geo):
    xx = [x['name'] for x in cov_data if x['penalizer'] == w_geo]
    yy = [x['Entropy'] for x in cov_data if x['penalizer'] == w_geo]
    fig.add_trace(
        go.Box(y=yy, x=xx, name=str(w_geo), marker_color=utils.color_palette[w])
    )
fig.update_xaxes(visible=False)
fig.update_yaxes(title_text='TIE')
fig.update_layout(
    boxmode='group',
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "right", 'x': 0.99},
)
fig.show()

fig = go.Figure()
names = [
    'Low penalization',
    'Medium penalization',
    'High penalization',
    'High penalization',
    'High penalization',
]
for w, w_geo in enumerate(ww_geo):
    xx = ['nothing' for x in cov_data if x['penalizer'] == w_geo]
    yy = [x['Entropy'] for x in cov_data if x['penalizer'] == w_geo]
    fig.add_trace(
        go.Box(y=yy, x=xx, name=names[w], marker_color=utils.color_palette[w])
    )
fig.update_xaxes(visible=False)
fig.update_yaxes(title_text='TIE')
fig.update_layout(
    boxmode='group',
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "right", 'x': 0.99},
)
fig.show()
