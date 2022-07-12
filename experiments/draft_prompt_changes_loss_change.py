import torch
import pydiffvg
from config import args
from pathlib import Path
from src import loss, utils
from drawing_model import DrawingModel
import src.fid_score as fid
import copy
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


with open('results/prompt_change/losses.pkl', 'rb') as f:
    df_a = pickle.load(f)
with open('results/prompt_change/losses2.pkl', 'rb') as f:
    df_b = pickle.load(f)

df = pd.concat([df_a, df_b])

PROMPT_TYPES = ['Old prompt', 'Changed prompt', 'New prompt']
COLORS = ['rgba(0,170,123,A)', 'rgba(0,83,170,A)', 'rgba(255,157,0,A)']

fig = go.Figure()


for pt, prompt_type in enumerate(PROMPT_TYPES):

    yy = []
    yy_upper = []
    yy_lower = []
    df2 = df[df['type'] == prompt_type]
    xx = list(range(min(df2['iter']), max(df2['iter'])))

    for t in xx:
        losses_t = df2[df2['iter'] == t]['loss'].to_numpy()
        mu_t = np.mean(losses_t)
        sd_t = np.std(losses_t)
        yy.append(mu_t)
        yy_upper.append(mu_t + sd_t)
        yy_lower.append(mu_t - sd_t)

    fig.add_trace(
        go.Scatter(
            x=xx + xx[::-1],
            y=yy_upper + yy_lower[::-1],
            fill='toself',
            fillcolor=COLORS[pt].replace('A', '0.5'),
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name=prompt_type,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=yy,
            line_color=COLORS[pt].replace('A', '1'),
            name=prompt_type,
        )
    )
fig.update_traces(mode='lines')
fig.update_layout(
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "right", 'x': 0.99},
    xaxis_title="iteration",
    yaxis_title="semantic loss w.r.t. new prompt",
)
fig.show()

from scipy import stats

df_box = df[df['iter'] == max(df['iter']) - 2]

a = df_box[df_box['type'] == 'Changed prompt']['loss'].to_numpy()
b = df_box[df_box['type'] == 'New prompt']['loss'].to_numpy()

print(stats.ttest_ind(a, b))

fig = px.box(df_box, x="name", y="loss", color="type")
fig.show()
