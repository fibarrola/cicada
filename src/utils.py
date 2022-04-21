import torch
import numpy as np


def save_data(time_str, params):
    with open('results/' + time_str + '.txt', 'w') as f:
        f.write('I0: ' + params.svg_path + '\n')
        f.write('prompt: ' + str(params.clip_prompt) + '\n')
        f.write('num paths: ' + str(params.num_paths) + '\n')
        f.write('num_iter: ' + str(params.num_iter) + '\n')
        f.write('w_points: ' + str(params.w_points) + '\n')
        f.write('w_colors: ' + str(params.w_colors) + '\n')
        f.write('w_widths: ' + str(params.w_widths) + '\n')
        f.write('w_img: ' + str(params.w_img) + '\n')
        f.close()


def area_mask(width, height, drawing_area):
    j0 = round(drawing_area['x0'] * width)
    j1 = round(drawing_area['x1'] * width)
    i0 = round((1 - drawing_area['y1']) * height)
    i1 = round((1 - drawing_area['y0']) * height)
    mask = torch.ones((height, width, 3))
    mask[i0:i1, j0:j1, :] = torch.zeros((i1 - i0, j1 - j0, 3))
    mask = mask[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)
    return mask


def get_nouns():
    with open('data/nouns.txt', 'r') as f:
        nouns = f.readline()
        f.close()
    nouns = nouns.split(" ")
    noun_prompts = ["a drawing of a " + x for x in nouns]
    return nouns, noun_prompts


# Print iterations progress
def printProgressBar(
    iteration,
    total,
    loss,
    prefix='Progress',
    suffix='-- Loss: ',
    decimals=1,
    length=50,
    fill='█',
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    loss = "{:3f}".format(loss)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} {loss}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def k_min_elements(X, K):
    return np.argsort(X)[:K]


def k_max_elements(X, K):
    return np.argsort(X)[K:]
