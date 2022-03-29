import torch
import random
import pydiffvg
from svgpathtools import svg2paths
from xml.dom import minidom


class DrawingPath():
    def __init__(self, path, color, width, num_segments):
        self.path = path
        self.color = color
        self.width = width
        self.num_segments = num_segments

def get_width(line):
    width_idx_0 = prop_line.find('stroke-width:')
    width_idx_1 = min( prop_line[width_idx_0+13:].find(';'), prop_line[width_idx_0+13:].find('p') )
    width = float(prop_line[width_idx_0+13:width_idx_0+13+width_idx_1])
    width = torch.tensor(width)

def get_drawing_paths(path_to_svg_file):
    path_list = []
    paths, attributes = svg2paths(path_to_svg_file)
    
    for att in attributes:
        style = att['style'].split(';')
        for x in style:
            if len(x) >= 13:
                if x[:13] == 'stroke-width:':
                    width = float(x[13:])
            if len(x) >= 15:
                if x[:15] == 'stroke-opacity:':
                    opacity = float(x[15:])
            if len(x) >= 8:
                if x[:8] == 'stroke:#':
                    hex_code = x[8:]
                    color = list(int(hex_code[i:i+2], 16)/255 for i in (0, 2, 4))
        color.append(opacity)
        color = torch.tensor(color)
        width = torch.tensor(width)

        try:
            [x_a, x_b] = att['d'].split('c')
        except:
            [x_a, x_b] = att['d'].split('C')
        x0 = [float(x) for x in x_a[2:].split(',')]
        points = [xx.split(',') for xx in x_b[1:].split(' ')]
        points = [[float(x), float(y)] for [x,y] in points] 
        path = [x0]+points
        num_segments = len(path)//3
        path = torch.tensor(path)
        v0 = torch.tensor([0,0])
        for k in range(path.size(0)):
            path[k,:] += v0
            if k%3 == 0:
                v0 = path[k,:]

    
        path_list.append(DrawingPath(path, color, width, num_segments))

    # print(path, color, width, num_segments)    
    return path_list

# def get_drawing_paths(path_to_svg_file):
    # f =  open(path_to_svg_file, 'r')
    # lines = f.readlines()
    # f.close()
    # path_list = []
    
    # for l, line in enumerate(lines):
    #     if len(line)<9:
    #         continue
    #     if line[:9] == '    <path':
    #         for k in range(1,4):
    #             if lines[l+k][7:12] == 'style':
    #                 prop_line = lines[l+k]
    #             elif lines[l+k][7] == 'd':
    #                 path_line = lines[l+k]
            
    #         # Get color
    #         color_idx_0 = prop_line.find('stroke:#')
    #         color_idx_1 = prop_line[color_idx_0+8:].find(';')
    #         opacity_idx_0 = prop_line.find('stroke-opacity:')
    #         opacity_idx_1 = min( [prop_line[opacity_idx_0+15:].find('"'), prop_line[opacity_idx_0+15:].find(';')])
    #         hex = prop_line[color_idx_0+8:color_idx_0+8+color_idx_1]
    #         color = list(int(hex[i:i+2], 16)/255 for i in (0, 2, 4))
    #         color.append(float(prop_line[opacity_idx_0+15:opacity_idx_0+15+opacity_idx_1]))
    #         color = torch.tensor(color)
    #         # Get width
    #         width_idx_0 = prop_line.find('stroke-width:')
    #         width_idx_1 = prop_line[width_idx_0+13:].find(';')
    #         width = float(prop_line[width_idx_0+13:width_idx_0+13+width_idx_1])
    #         width = torch.tensor(width)
    #         # Get path
    #         idx0 = path_line.find('m')
    #         idx1 = path_line.find('c')
    #         path = [[float(x) for x in path_line[idx0+2:idx1-1].split(',')]]
    #         linedata = path_line[idx1+2:-2].split(' ')
    #         for k in range(len(linedata)):
    #             path.append([float(x) for x in linedata[k].split(',')])
    #         num_segments = len(path)//3
    #         path = torch.tensor(path)
    #         v0 = torch.tensor([0,0])
    #         for k in range(path.size(0)):
    #             path[k,:] += v0
    #             if k%3 == 0:
    #                 v0 = path[k,:]
                        
    #         path_list.append(DrawingPath(path, color, width, num_segments))
    # print(path, color, width, num_segments)

    # dsasd
    # return path_list


def treebranch_initialization(path_list, num_paths, canvas_width, canvas_height, x0, x1, y0, y1):

    # Get all endpoints within drawing region
    starting_points = []
    for path in path_list:
        for k in range(path.path.size(0)):
            if k % 3 == 0:
                if (x0 < path.path[k,0] < x1) and (y0 < (1-path.path[k,1]) < y1):
                    starting_points.append(tuple([x.item() for x in path.path[k]]))

    # We build 3 groups:
    # 1) Curves starting from existing endpoints
    # 2) Curves starting from curves in 1)
    # 3) Random curves
    K1 = num_paths//4

    # Initialize Curves
    shapes = []
    shape_groups = []
    first_endpoints = []

    # Add random curves
    for k in range(num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        points = []
        if k < K1:
            p0 = random.choice(starting_points)
        elif k < 2*K1:
            p0 = random.choice(first_endpoints)
        else:
            p0 = (random.random()*(x1-x0)+x0, random.random()*(y1-y0)+1-y1)
        points.append(p0)

        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3

        if k < K1:
            first_endpoints.append(points[-1])

        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points = num_control_points, points = points, stroke_width = torch.tensor(1.0), is_closed = False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
        shape_groups.append(path_group)

    return shapes, shape_groups



    
        