import torch

class DrawingPath():
    def __init__(self, path, color, width, num_segments):
        self.path = path
        self.color = color
        self.width = width
        self.num_segments = num_segments


def get_drawing_paths(path_to_svg_file):
    f =  open(path_to_svg_file, 'r')
    lines = f.readlines()
    f.close()

    path_list = []

    for l, line in enumerate(lines):
        if len(line)<9:
            continue
        if line[:9] == '    <path':
            prop_line = lines[l+1]
            path_line = lines[l+2]
            # Get color
            color_idx_0 = prop_line.find('stroke:#')
            color_idx_1 = prop_line[color_idx_0+8:].find(';')
            opacity_idx_0 = prop_line.find('stroke-opacity:')
            opacity_idx_1 = prop_line[opacity_idx_0+15:].find('"')
            hex = prop_line[color_idx_0+8:color_idx_0+8+color_idx_1]
            color = list(int(hex[i:i+2], 16)/255 for i in (0, 2, 4))
            color.append(float(prop_line[opacity_idx_0+15:opacity_idx_0+15+opacity_idx_1]))
            color = torch.tensor(color)
            # Get width
            width_idx_0 = prop_line.find('stroke-width:')
            width_idx_1 = prop_line[width_idx_0+13:].find(';')
            width = float(prop_line[width_idx_0+13:width_idx_0+13+width_idx_1])
            width = torch.tensor(width)
            # Get path
            idx0 = path_line.find('m')
            idx1 = path_line.find('c')
            path = [[float(x) for x in path_line[idx0+2:idx1-1].split(',')]]
            linedata = path_line[idx1+2:-2].split(' ')
            for k in range(len(linedata)):
                path.append([float(x) for x in linedata[k].split(',')])
            num_segments = len(path)//3
            path = torch.tensor(path)
            v0 = torch.tensor([0,0])
            for k in range(path.size(0)):
                path[k,:] += v0
                if k%3 == 0:
                    v0 = path[k,:]
            
            path_list.append(DrawingPath(path, color, width, num_segments))

    return path_list