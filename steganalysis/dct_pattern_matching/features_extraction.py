import numpy as np
from scipy.fftpack import dct 
from PIL import Image 
from rich.progress import track 


Y_quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

zigzag_order = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]

def zigzag_scan(block):
    return block.flatten()[zigzag_order]

def apply_dct_and_quantize(image, block_size=8):
    """Apply DCT to the image and quantize using the standard JPEG quantization tables."""
    h, w = image.shape
    quantized_dct = np.zeros_like(image, dtype=float)

    # Apply DCT and quantize block-by-block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Extract the 8x8 block
            block = image[i:i+block_size, j:j+block_size]
            
            # Apply 2D DCT (first along rows, then along columns)
            block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Quantize the DCT coefficients using the Y quantization table
            block_quantized = np.round(block_dct / Y_quantization_table)  # You can use C_quantization_table for chroma channels
            
            # Store the quantized DCT coefficients
            quantized_dct[i:i+block_size, j:j+block_size] = block_quantized

    return quantized_dct

def apply_DCT(img_arr, block_size = 8):
    # Apply DCT and quantization
    quantized_dct = apply_dct_and_quantize(img_arr)

    # Zig-zag scan the quantized DCT coefficients
    zigzagged_dct = np.zeros_like(quantized_dct, dtype=int)

    h, w = quantized_dct.shape

    # Apply zig-zag scan to all 8x8 blocks
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = quantized_dct[i:i+block_size, j:j+block_size]
            zigzagged_dct[i:i+block_size, j:j+block_size] = np.array(zigzag_scan(block)).reshape((block_size, block_size))

    return zigzagged_dct

def pattern_counting(dct_array, S = 4):
    H, W = dct_array.shape
    T_arr = np.zeros(S ** 4, dtype=int)
    for h in range(2, H):
        for w in range(0, W-1):
            cof_neigh = np.zeros(5, dtype=int)
            cof_neigh[0] = dct_array[h - 2, w]
            cof_neigh[1] = dct_array[h - 2, w + 1]
            cof_neigh[2] = dct_array[h - 1, w]
            cof_neigh[3] = dct_array[h - 1, w + 1]
            cof_neigh[4] = dct_array[h, w + 1]

            ref_cof_min = np.min(cof_neigh)
            ref_cof_max = np.max(cof_neigh)
            
            for ref_cof in [ref_cof_max, ref_cof_min]:
                ref_idx = np.where(cof_neigh == ref_cof)[0][0]
                l, ul, ur, r = 0, 0, 0, 0
                match ref_idx:
                    case 0:
                        l = cof_neigh[1]
                        ul = cof_neigh[3]
                        ur = cof_neigh[4]
                        r = cof_neigh[2]
                    case 1:
                        l = cof_neigh[3]
                        ul = cof_neigh[2]
                        ur = cof_neigh[4]
                        r = cof_neigh[0]   
                    case 2:
                        l = cof_neigh[0]
                        ul = cof_neigh[1]
                        ur = cof_neigh[3]
                        r = cof_neigh[4]
                    case 3:
                        l = cof_neigh[4]
                        ul = cof_neigh[2]
                        ur = cof_neigh[0]
                        r = cof_neigh[1]
                    case 4:
                        l = cof_neigh[2]
                        ul = cof_neigh[0]
                        ur = cof_neigh[1]
                        r = cof_neigh[3] 
                
                P_arr = [min(np.abs(l - ref_cof), S - 1),
                         min(np.abs(ul - ref_cof), S - 1), 
                         min(np.abs(ur - ref_cof), S - 1), 
                         min(np.abs(r - ref_cof), S - 1)]
                M = P_arr[0] * S ** 3 + P_arr[1] * S ** 2
                M += P_arr[2] * S + P_arr[3]    #Na razie nie dodaje +1 bo idx od 0
                
                T_arr[M] += 1

    return T_arr 

'''
def get_cropped_img(img_arr, crop_edge = 4):
    crop_img_arr = np.copy(img_arr)
    #print(crop_img_arr)
    #H, W = img_arr.shape
    crop_img_arr[:crop_edge, :] = 0
    crop_img_arr[-crop_edge, :] = 0
    crop_img_arr[:, :crop_edge] = 0
    crop_img_arr[:, -crop_edge:] = 0
    
    return crop_img_arr
'''

def get_cropped_img(img_arr, crop_edge = 4):
    return img_arr[crop_edge:-crop_edge,
                   crop_edge:-crop_edge]

def get_features_vect(t_arr, t_prim_arr):
    feat_vect = np.zeros(t_arr.shape[0], dtype = float)
    for i in range(t_arr.shape[0]):
        if t_prim_arr[i] == 0.:
            if t_arr[i] == 0.:
                feat_vect[i] = 1.
            else:
                feat_vect[i] = -1.
        else:            
            feat_vect[i] = t_arr[i] / t_prim_arr[i]
    feat_min = np.max(np.min(feat_vect), 0)
    feat_max = np.max(feat_vect)
    
    for i in range(feat_vect.shape[0]):
        if feat_vect[i] < 0:
            feat_vect[i] = feat_max

    feat_vect = (feat_vect - feat_min)
    feat_vect /= (feat_max - feat_min)

    return feat_vect 

def get_img_features(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    img_arr = np.array(img)
    img_c_arr = get_cropped_img(img_arr)
    
    img_dct = apply_DCT(img_arr)
    img_c_dct = apply_DCT(img_c_arr)

    t = pattern_counting(img_dct)
    t_c = pattern_counting(img_c_dct)

    return get_features_vect(t, t_c) 


img_path = 'data/stego_png/'
file = open("stego_training_data.txt", "w")
# 0 - cover class, 1 - stego class

for i in track(range(500)):
    feat_vect = get_img_features(img_path + str(i) + '.png')
    for f in feat_vect:
        file.write(str(f) + ",")
    file.write("1\n")

file.close()