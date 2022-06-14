import numpy as np
import pandas as pd
import src.fid_score as fid
import plotly.express as px
import plotly.graph_objects as go

names = ['chair', 'hat', 'lamp', 'pot']

fid_data = []
for n, name in enumerate(names):
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
            f"results/fix_paths3/{name}/{process_name}", rand_sampled_set_dim=subset_dim
        )
        gen_type = 'standard' if process_name == 'from_scratch' else 'trace-conditioned'
        fid_data.append(
            {
                'Covariance Norm': np.linalg.norm(S),
                'name': name,
                'generation': gen_type,
            }
        )

import pickle

with open('results/fix_paths3/data.pkl', 'wb') as f:
    pickle.dump(fid_data, f)

df = pd.DataFrame(fid_data)

fig = px.scatter(
    df,
    x="name",
    y="Covariance Norm",
    color="generation",  # size=[2 for x in range(len(df))]
)
fig.show()

fig = px.box(
    df,
    x="name",
    y="Covariance Norm",
    color="generation",  # size=[2 for x in range(len(df))]
)
fig.show()

fig = go.Figure()


xx = df.query('generation=="standard"')['name']
yy = df.query('generation=="standard"')['Covariance Norm']
fig.add_trace(go.Box(y=yy, x=xx, name='standard', marker_color='rgba(255,157,0,1)'))

xx = df.query('generation=="trace-conditioned"')['name']
yy = df.query('generation=="trace-conditioned"')['Covariance Norm']
fig.add_trace(
    go.Box(y=yy, x=xx, name='trace-conditioned', marker_color='rgba(0,83,170,1)')
)

fig.update_layout(
    boxmode='group',
    legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01},
)
fig.show()
