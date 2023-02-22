import os
import pickle
import torch
import pandas as pd
from mapelites_config import args
from drawing_model import Cicada
from behaviour import TextBehaviour

#
# Preliminaries
#
device = "cuda:0" if torch.cuda.is_available() else "cpu"
os.makedirs(args.save_path, exist_ok=True)
text_behaviour = TextBehaviour()
behaviour_dims =[x.split("|") for x in args.behaviour_dims.split("||")]
for bd in behaviour_dims:
    text_behaviour.add_behaviour(bd[0], bd[1])
df = pd.DataFrame(columns=["fintess"]+[beh["name"] for beh in text_behaviour.behaviours])

#
# Aux
#
def run_cicada(cicada, args):
    cicada.set_penalizers(
        w_points=args.w_points,
        w_colors=args.w_colors,
        w_widths=args.w_widths,
        w_geo=args.w_geo,
    )
    cicada.process_text(args.prompt)
    cicada.load_svg_shapes(args.svg_path)
    cicada.add_random_shapes(args.num_paths)
    cicada.initialize_variables()
    cicada.initialize_optimizer()
    losses = []
    behs = []
    for t in range(args.num_iter):
        cicada.run_epoch()
        if t > args.num_iter-11:
            with torch.no_grad():
                losses.append(cicada.losses["global"].detach())
                behs.append(text_behaviour.eval_behaviours(cicada.img))
    
    loss = torch.mean(torch.cat(losses)).item()
    behs = torch.mean(torch.cat([b.unsqueeze(0) for b in behs]), dim=0)
    data = [1-loss]+[b.item() for b in behs]
    return data, cicada.drawing

#
#
# MAPELITES
#
#

#
# Generate population
#
for k in range(args.population_size):
    cicada = Cicada(
        device=device,
        drawing_area=args.drawing_area,
        max_width=args.max_width,
    )
    
    data, drawing = run_cicada(cicada, args)
    df.loc[drawing.id] = data
    with open(f"{args.save_path}/{drawing.id}.pkl", "wb") as f:
        pickle.dump(drawing, f)
    df.to_csv(f"{args.save_path}/df.csv")
