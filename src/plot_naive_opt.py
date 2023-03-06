import pandas as pd
import plotly.graph_objects as go


for save_path in [
    "results/naive/chair_1",
    "results/naive/dress_2",
]:
    df = pd.read_csv(f"{save_path}/df.csv")
    print(df.columns.values)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[df.columns.values[-2]],
            y=df[df.columns.values[-1]],
            mode='markers',
        )
    )
    fig.update_layout(
        title=save_path
    )
    fig.show()