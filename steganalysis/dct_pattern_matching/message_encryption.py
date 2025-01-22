import numpy as np 
from PIL import Image 

''' 
Encrypting text message into .png image 
'''

def encrypt_message(img_path, message):
    # Geting img RGB array (uint8)
    img = Image.open(img_path)
    if '.png' in img_path:
        img = img.convert('RGB')
    img = np.array(img)

    #start_flag = '1111110'
    end_flag = '11111100'
    message = message + chr(int(end_flag, base=2)) 
    rows, cols, *_ = img.shape
    if len(message) + 2 > (rows * cols * 3) // 8:
        return 'The message is to long to be encrypted in the image'
    
    idx = 0
    img = img.flatten()
    for i in range(0, len(message) * 8, 8):
        char = format(ord(message[idx]), '08b')
        for j, c in enumerate(char):
            if c == '0':
                if img[i + j] % 2 == 1:
                    img[i + j] += 1
            else:
                if img[i + j] % 2 == 0:
                    img[i + j] += 1
            img[i + j] %= 255 
        idx += 1 

    img = img.reshape(rows, cols, 3) 
    return img  



def decrypt_message(img_path):
    img = Image.open(img_path)
    img = np.array(img) 

    #start_flag = '11111110'
    end_flag = '11111100'

    #start_encountered = True
    message = ''

    img = img.flatten()
    img = [b % 2 for b in img]
    
    idx = 0
    while True:
        byte = ''.join(map(str, img[idx * 8 : idx * 8 + 8]))
        #print(byte)
        if byte == end_flag:
            #print('wyszlo')
            break 
        message += chr(int(byte, base=2))
        idx += 1
                    
    return message
