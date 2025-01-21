import os
import numpy as np
from PIL import Image, ImageOps

def add_noise(image, noise_level=20):
    """
    Dodaje szum do obrazu czarno-białego.
    
    :param image: Obiekt PIL.Image w trybie L (czarno-biały).
    :param noise_level: Intensywność szumu (0-255).
    :return: Obraz z dodanym szumem.
    """
    np_image = np.array(image, dtype=np.int16)
    noise = np.random.randint(-noise_level, noise_level, np_image.shape, dtype=np.int16)
    np_image = np.clip(np_image + noise, 0, 255)  # Dodanie szumu i ograniczenie wartości do zakresu 0-255
    return Image.fromarray(np_image.astype(np.uint8), mode="L")

def compress_png_images(input_folder, output_folder, noise_level=0, quality=95):
    """
    Kompresuje pliki PNG z możliwością dodania szumu i zapisuje je w nowym folderze.

    :param input_folder: Ścieżka do folderu z obrazami PNG do kompresji.
    :param output_folder: Ścieżka do folderu, gdzie zostaną zapisane skompresowane obrazy.
    :param noise_level: Intensywność szumu do dodania (0 = brak szumu).
    :param quality: Poziom jakości (1-100), domyślnie 95.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(input_path) as img:
                if img.mode != "L":
                    img = img.convert("L")
                
                img = add_noise(img, noise_level=20)
                img.save(output_path, format="PNG", optimize=True)

            print(f"Skompresowano: {filename} i zapisano do {output_folder}")

if __name__ == "__main__":
    
    input_folder = "./bossbase_data/cover"  # Ścieżka do folderu z obrazami PNG
    output_folder = "./bossbase_data/cover_compressed"

    compress_png_images(input_folder, output_folder)
