import numpy as np
import pandas as pd
import src.fid_score as fid
import plotly.express as px

names = ['chair', 'hat', 'lamp', 'pot']

fid_data = []
for n, name in enumerate(names):
    for process_name in ['from_scratch','from_scratch','from_scratch','from_scratch', 'from_scratch', 'gen_trial0', 'gen_trial1', 'gen_trial100', 'gen_trial101', 'gen_trial102']:
        subset_dim = 20 if process_name == 'from_scratch' else None
        mu, S = fid.get_statistics(f"results/fix_paths/{name}/{process_name}", rand_sampled_set_dim=subset_dim)
        gen_type = 'standard' if process_name == 'from_scratch' else 'trace-conditioned'
        fid_data.append(
            {'Covariance Norm': np.linalg.norm(S), 'name': name, 'generation': gen_type,}
        )

import pickle
with open('results/fix_paths/data.pkl', 'wb') as f:
    pickle.dump(fid_data, f)

df = pd.DataFrame(fid_data)

fig = px.scatter(
    df, x="name", y="Covariance Norm", color="generation", #size=[2 for x in range(len(df))]
)
fig.show()
