import math
import numpy as np
from PIL import Image

from helpers import entropy


def dct(X:np.ndarray, N:int = 16) -> np.ndarray:
    """
    # Returns:
    `np.ndarray` of size (-1, `N`, `N`)
    """
    # source: https://dev.to/marycheung021213/understanding-dct-and-quantization-in-jpeg-compression-1col
    DCT=np.zeros((N,N))
    for m in range(N):
        for n in range(N):
            if m==0:
                DCT[m][n]=math.sqrt(1/N)
            else:
                DCT[m][n]=math.sqrt(2/N)*math.cos((2*n+1)*math.pi*m/(2*N))

    y_size, x_size = X.shape
    X = np.pad(X, constant_values=0, pad_width=((0, (N-y_size%N)%N), (0, (N-x_size%N)%N)))
    y_size, x_size = X.shape

    X_dct = np.zeros_like(X)

    for i in range(0, y_size, N):
        for j in range(0, x_size, N):
            # get DCT coefficients (DCT 2nd type)
            X_dct[i:i+N, j:j+N] = DCT @ X[i:i+N, j:j+N] @ DCT.T
    
    blocks = X_dct.reshape(-1, N, N)

    return blocks

def features_from_image(image:np.ndarray|Image.Image):
    X_dct = dct(np.array(image))
    X = []
    for block in X_dct:
        X.append(np.mean(block))
        X.append(entropy(block))
        for i in range(1, 2): # high freqs
            for j in range(1, 2):
                X.append(block[-i, -j])
    return X