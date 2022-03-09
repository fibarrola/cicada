import torch

def save_data(time_str, params):
    with open('results/'+time_str+'.txt', 'w') as f:
        f.write('I0: ' +params.svg_path +'\n')
        f.write('prompt: ' +str(params.clip_prompt) +'\n')
        f.write('num paths: ' +str(params.num_paths) +'\n')
        f.write('num_iter: ' +str(params.num_iter) +'\n')
        f.write('w_points: '+str(params.w_points)+'\n')
        f.write('w_colors: '+str(params.w_colors)+'\n')
        f.write('w_widths: '+str(params.w_widths)+'\n')
        f.write('w_img: '+str(params.w_img)+'\n')
        f.close()


def area_mask(width, height, x0=0, x1=1, y0=0, y1=1):
    j0 = round(x0*width)
    j1 = round(x1*width)
    i0 = round((1-y1)*height)
    i1 = round((1-y0)*height)
    mask = torch.ones((height, width,3))
    mask[i0:i1, j0:j1, :] = torch.zeros((i1-i0, j1-j0, 3))
    return mask



