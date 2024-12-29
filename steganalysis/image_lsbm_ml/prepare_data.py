import os
import random
import numpy as np
from PIL import Image

def embedd_random_msg(image:Image, usage:float|int):
    assert 0 <= usage <= 1

    image:np.ndarray = np.array(image)
    height, width = image.shape
    msg = np.random.randint(2, size=int(height * width * usage))

    indices = np.random.choice(height * width, len(msg), replace=False)
    positions = np.unravel_index(indices, (height, width))

    for pos, bit in zip(zip(*positions), msg):
        grey_byte = image[pos]
        # Embed the bit if it doesn't already match
        if grey_byte & 1 != bit:
            if grey_byte == 255:  # Avoid overflow
                grey_byte = 254
            elif grey_byte == 0:  # Avoid underflow
                grey_byte = 1
            else:
                if random.random() < 0.5:
                    grey_byte -= 1
                else:
                    grey_byte += 1

        image[pos] = grey_byte

    return Image.fromarray(image)

def prepare_data(usage, how_many:int = 10001):
    assert how_many <= 10001
    path = os.getcwd() + f'/data/stego_{int(usage*100)}/'
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(1, how_many):
        image = Image.open(f"data/cover/{i}.png")
        new_image = embedd_random_msg(image, usage)
        new_image.save(f"data/stego_{int(usage*100)}/{i}_stego.png")