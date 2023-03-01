import os
import pickle
import torch
import random
import shortuuid
import pydiffvg
import pandas as pd
import src.fid_score as fid
import numpy as np
import plotly.graph_objects as go
from utils import tie
from mapelites_config import args
from drawing_model import Cicada
from behaviour import TextBehaviour

#
# Preliminaries
#
device = "cuda:0" if torch.cuda.is_available() else "cpu"

k = 0
while os.path.exists(f"{args.save_path}_{k}"):
    k += 1
save_path = f"{args.save_path}_{k}"
os.makedirs(save_path)
os.makedirs(save_path + "/initial_population")
os.makedirs(save_path + "/final_population")
delattr(args, "save_path")

text_behaviour = TextBehaviour()
behaviour_dims = [x.split("|") for x in args.behaviour_dims.split("||")]
for bd in behaviour_dims:
    text_behaviour.add_behaviour(bd[0], bd[1])
df = pd.DataFrame(
    columns=["in_population", "orig_iter", "fitness"]
    + [beh["name"] for beh in text_behaviour.behaviours]
)


#
# Aux
#
def run_cicada(args, drawing=None, mutate=False, num_iter=1000):
    cicada = Cicada(
        device=device,
        drawing_area=args.drawing_area,
        max_width=args.max_width,
        drawing=drawing,
    )
    cicada.set_penalizers(
        w_points=args.w_points,
        w_colors=args.w_colors,
        w_widths=args.w_widths,
        w_geo=args.w_geo,
    )
    cicada.process_text(args.prompt)
    if not mutate:
        cicada.load_svg_shapes(args.svg_path)
        cicada.add_random_shapes(args.num_paths)
    cicada.initialize_variables()
    cicada.initialize_optimizer()
    if mutate:
        cicada.mutate_respawn_traces()
    losses = []
    behs = []
    for t in range(num_iter):
        cicada.run_epoch()
        if t > num_iter - 11:
            with torch.no_grad():
                losses.append(cicada.losses["global"].detach())
                behs.append(text_behaviour.eval_behaviours(cicada.img))

    loss = torch.mean(torch.cat(losses)).item()
    behs = torch.mean(torch.cat([b.unsqueeze(0) for b in behs]), dim=0)
    fitness = 1 - loss
    behs = [b.item() for b in behs]
    return fitness, behs, cicada.drawing


class Grid:
    def __init__(self):
        self.id_mat = None
        self.fit_mat = -10.0
        self.dims = {}

    def add_scale(self, dim_name, value_list, num_slots):
        mx = min(value_list)
        Mx = max(value_list)
        grid_min = mx - 0.1 * (Mx - mx)
        grid_max = Mx + 0.1 * (Mx - mx)
        values = [
            grid_min + k * (grid_max - grid_min) / (num_slots - 2)
            for k in range(num_slots - 1)
        ]
        self.dims[dim_name] = values
        self.id_mat = np.array([self.id_mat for k in range(num_slots)])
        self.fit_mat = np.array([self.fit_mat for k in range(num_slots)])

    def get_grid_idx(self, beh, dim_name):
        grid_idx = 0
        for value in self.dims[dim_name]:
            if beh < value:
                break
            else:
                grid_idx += 1

        return grid_idx

    def allocate(self, id, behs, fitness):
        grid_idx = []
        for d, dim_name in enumerate(self.dims):
            grid_idx.append(self.get_grid_idx(behs[d], dim_name))

        grid_idx = tuple(grid_idx)
        if fitness > self.fit_mat[grid_idx]:
            self.fit_mat[grid_idx] = fitness
            replaced_id = self.id_mat[grid_idx]
            self.id_mat[grid_idx] = id
            return True, replaced_id

        return False, None

    def image_array_2d(self, save_path, name):
        assert len(self.id_mat.shape) == 2
        for i in range(self.id_mat.shape[0]):
            for j in range(self.id_mat.shape[1]):
                if self.id_mat[i, j] is None:
                    img = torch.ones((224, 224, 3), device="cpu", requires_grad=False)
                else:
                    with open(f"{save_path}/{self.id_mat[i,j]}.pkl", "rb") as f:
                        drawing = pickle.load(f)
                    drawing.render_img()
                    img = drawing.img.cpu().permute(0, 2, 3, 1).squeeze(0)
                pydiffvg.imwrite(
                    img, f"{save_path}/{name}/{i}{j}.png", gamma=1,
                )


# Generate population
for k in range(args.population_size):
    print(f"Building {k}-th initial individual...")
    fitness, behs, drawing = run_cicada(args, num_iter=args.num_iter)
    df.loc[drawing.id] = [False, 0, fitness] + behs
    with open(f"{save_path}/{drawing.id}.pkl", "wb") as f:
        pickle.dump(drawing, f)
    df.to_csv(f"{save_path}/df.csv", index_label="id")

# Build grid
grid = Grid()
for beh in text_behaviour.behaviours:
    grid.add_scale(beh['name'], list(df[beh['name']]), args.grid_size)

# Fill grid
for id in df.index:
    x = df.loc[id]
    behs = [x[dim_name] for dim_name in grid.dims]
    in_population, replaced_id = grid.allocate(id, behs, x['fitness'])
    df.at[id, "in_population"] = in_population
    if replaced_id is not None:
        df.at[replaced_id, "in_population"] = False

df.to_csv(f"{save_path}/df.csv", index_label="id")

grid.image_array_2d(save_path, "initial_population")
# mu, S = fid.get_statistics(
#     f"{save_path}/initial_population",
#     rand_sampled_set_dim=10,
# )
print("initial Population")
print(df)
print(grid.id_mat)
print(grid.fit_mat)
# print('Entropy: ', tie(S))
print("")

fig = go.Figure()
filtered_df = df
fig.add_trace(
    go.Scatter(
        x=filtered_df[text_behaviour.behaviours[0]["name"]],
        y=filtered_df[text_behaviour.behaviours[1]["name"]],
        mode='markers',
        name="Initial population",
    )
)


# Search
for iter in range(args.mapelites_iters):
    print(f"Building {iter}-th mutant...")
    mutant_id = random.choice(df[df["in_population"] == True].index)
    with open(f"{save_path}/{mutant_id}.pkl", "rb") as f:
        drawing = pickle.load(f)
    drawing.id = shortuuid.uuid()
    fitness, behs, drawing = run_cicada(
        args, drawing=drawing, mutate=True, num_iter=args.num_iter // 2
    )
    in_population, replaced_id = grid.allocate(drawing.id, behs, fitness)
    df.loc[drawing.id] = [in_population, iter + 1, fitness] + behs
    if replaced_id is not None:
        df.at[replaced_id, "in_population"] = False

    # save
    with open(f"{save_path}/{drawing.id}.pkl", "wb") as f:
        pickle.dump(drawing, f)
    with open(f"{save_path}/grids.pkl", "wb") as f:
        pickle.dump(grid, f)
    df.to_csv(f"{save_path}/df.csv", index_label="id")

grid.image_array_2d(save_path, "final_population")
# mu, S = fid.get_statistics(
#     f"{save_path}/final_population",
#     rand_sampled_set_dim=10,
# )
print("Final Population")
print(df)
print(grid.id_mat)
print(grid.fit_mat)
# print('Entropy: ', tie(S))
print("")


filtered_df = df[df["in_population"] == True]
fig.add_trace(
    go.Scatter(
        x=filtered_df[text_behaviour.behaviours[0]["name"]],
        y=filtered_df[text_behaviour.behaviours[1]["name"]],
        mode='markers',
        name="Final population",
    )
)
fig.show()
