import math
import random
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def show_image(data, title=None, vmin=0, vmax=255):
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.facecolor'] = 'grey'   
    plt.gray()
    plt.axis('off')
    if title is not None:
        plt.title(title, fontsize=22)
    plt.imshow(data, vmin=vmin, vmax=vmax)

def show_grey_hist(data, title=None):
    plt.hist(np.array(data).flatten(), bins=255, color='grey')
    plt.xticks(range(0, 255, 10))
    plt.ylim(0, 4000)
    if title is not None:
        plt.title(title, fontsize=22)

def image_pair_gen(usage, max:int = None):
    i = 1
    if  max == None or max > 10001:
        max = 10001
    while i < max:
        idx = random.randint(1, 10000)
        cover = np.array(Image.open(f"data/cover/{idx}.png"))
        stego = np.array(Image.open(f"data/stego_{int(usage*100)}/{idx}_stego.png"))
        i +=1  
        yield cover, stego

def entropy(data):
    dens, _ = np.histogram(data, bins=256)
    dens = dens[np.flatnonzero(dens)]
    dens = dens / np.sum(dens)
    
    return -np.sum(dens * np.log(dens))