from PIL import Image, ImageDraw, ImageFont
from MTM import MTM_PWC
import numpy as np


image_fn = r'./data/input.png'
patch_fn = r'./data/patch.png'
alpha = [0, 51, 102, 153, 204, 256]

image = Image.open(image_fn).convert('LA')
patch = Image.open(patch_fn).convert('LA')

D = MTM_PWC(image=image, patch=patch, alpha=alpha)

min_loc = np.where(D == np.min(D))
D_vis = Image.fromarray(np.uint8(255*D))

font = ImageFont.truetype("arial.ttf", 25)
D_vis = D_vis.convert('RGB')
image = image.convert('RGB')
draw_D = ImageDraw.Draw(D_vis)
draw_image = ImageDraw.Draw(image)
draw_image.rectangle([(min_loc[1][0] - 0.5*patch.size[0], min_loc[0][0] - 0.5*patch.size[1]), (min_loc[1][0] + 0.5*patch.size[0], min_loc[0][0] + 0.5*patch.size[1])]
                     , outline="red")

draw_D.text((10, 10), 'Distance map', fill="red", font=font)
draw_image.text((10, 10), 'Best patch match', fill="red", font=font)

patch.show()
D_vis.show()
image.show()

