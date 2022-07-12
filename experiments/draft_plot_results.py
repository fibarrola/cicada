from tokenize import group
import pandas as pd
import matplotlib.pyplot as plt


d = {
    'Variance': [
        17.247810900117834,
        13.715313577593578,
        8.500301736317345,
        26.304661291021123,
        19.18850954651869,
        18.33934961978392,
        17.9282655663147,
        18.20788795566556,
        16.272605716639106,
        23.215518349920515,
        13.442396277546374,
        20.22241687929176,
    ],
    'Object': [
        'chair',
        'chair',
        'chair',
        'hat',
        'hat',
        'hat',
        'lamp',
        'lamp',
        'lamp',
        'pot',
        'pot',
        'pot',
    ],
    'Type': [
        'standard',
        'conditioned',
        'conditioned',
        'standard',
        'conditioned',
        'conditioned',
        'standard',
        'conditioned',
        'conditioned',
        'standard',
        'conditioned',
        'conditioned',
    ],
}


df = pd.DataFrame(data=d)

# print(df)

# plt.scatter(x='Object', y='Variance', data=df, color='Type')

# plt.show()

import plotly.express as px

fig = px.scatter(
    df, x="Object", y="Variance", color="Type", size=[10 for x in range(len(df))]
)
fig.show()
