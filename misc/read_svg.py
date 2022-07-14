import json
import subprocess

from PIL import Image
# import cairo
# import rsvg



with open('logs.json', 'rb') as f:
    a = json.load(f)

# for x in a[0]:
#     print(x, a[0][x],'\n')

# for x in a[0]['recorded_data']:
#     print(x, a[0]['recorded_data'][x],'\n')

count = 0
for y in a:
    if y['recorded_data']['event_type'] == 'save-sketch':
        # print(y['recorded_data']['user_name'])
        with open(f'svgs/{count}.svg', 'w') as f:
            f.write(y['recorded_data']['data']['svg'])
        subprocess.run(f"inkscape svgs/{count}.svg -o tmp.png --export-background-opacity=1".split(" "))
        image = Image.open("tmp.png")
        image = image.resize((224,224))
        image.save(fp=f"pngs/{count}.png")   
        count += 1

# img = cairo.ImageSurface(cairo.FORMAT_ARGB32, 640,480)

# ctx = cairo.Context(img)

# ## handle = rsvg.Handle(<svg filename>)
# # or, for in memory SVG data:
# handle= rsvg.Handle(None, svg_string)

# handle.render_cairo(ctx)

# img.write_to_png("svg.png") 
 
# subprocess.run("inkscape drawings/0.svg -o drawings/4.png --export-background-opacity=1 --export-width=224".split(" "))
# 

#  