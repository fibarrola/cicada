import json
import subprocess

from PIL import Image

# import cairo
# import rsvg


# with open('logs.json', 'rb') as f:
#     a = json.load(f)

a = [json.loads(line) for line in open('logs.json', 'rb')]
# for x in a[0]:
#     print(x, a[0][x],'\n')

# for x in a[0]['recorded_data']:
#     print(x, a[0]['recorded_data'][x],'\n')

count = 0
for y in a:
    if True:  # y['recorded_data']['event_type'] == 'save-sketch':
        print(y['recorded_data']['user_name'])
        svg_string = y['recorded_data']['data']['svg']
        print(svg_string)
        with open(f'svgs/{count}.svg', 'w') as f:
            f.write(svg_string)
        # string_list = svg_string.split('><path')
        # print(string_list[1][:20] == ' d="M387.05,431c-0.2')
        # break
        subprocess.run(
            f"inkscape Documents/co_creative_draw/misc/svgs/{count}.svg -o Documents/co_creative_draw/misc/tmp.png --export-background-opacity=1".split(
                " "
            )
        )
        # image = Image.open("tmp.png")
        # image = image.resize((224,224))
        # image.save(fp=f"pngs/{count}.png")
        # count += 1

    break

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
