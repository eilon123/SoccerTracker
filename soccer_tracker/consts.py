from PIL import Image, ImageDraw
import numpy as np


# field_mask
width = 1920
height = 1080
polygon = [(0, 488), (730, 236), (width - 1, 300), (width - 1, height - 1), (0, height - 1)]
field_mask = Image.new('L', (width, height), 0)
ImageDraw.Draw(field_mask).polygon(polygon, outline=1, fill=1)
field_mask = np.array(field_mask)