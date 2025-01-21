import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Implementacja algorytmu z DOI:10.1007/978-3-319-07674-4_73
def detect_stego_image(cover_image, suspicious_image, threshold):
    # Step 1: Wczytaj obrazy i zamień je na wartości intensywności (grayscale)
    cover = np.array(Image.open(cover_image).convert('L')) / 255.0
    suspicious = np.array(Image.open(suspicious_image).convert('L')) / 255.0

    # Step 2: Oblicz histogramy obu obrazów
    cover_histogram, _ = np.histogram(cover.flatten(), bins=256, range=(0, 1))
    suspicious_histogram, _ = np.histogram(suspicious.flatten(), bins=256, range=(0, 1))

    # Step 3: Wykres histogramów i różnice
    plt.bar(range(256), cover_histogram, alpha=0.5, label='Cover Image Histogram')
    plt.bar(range(256), suspicious_histogram, alpha=0.5, label='Suspicious Image Histogram')
    plt.legend()
    plt.title("Histogram Comparison")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    #plt.show()

    # Step 4: Różnice histogramów
    differences = suspicious_histogram - cover_histogram

    # Wyświetlanie histogramu różnic
    plt.bar(range(256), differences, color='red', alpha=0.7, label='Histogram Differences')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend()
    plt.title("Histogram Differences")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Difference in Frequency")
    #plt.show()

    # Step 5: Zlicz różnice z przeciwnym znakiem i tą samą wartością (z tolerancją)
    counter = 0
    tolerance = 1e-6  # Tolerancja dla numerycznych błędów
    for i in range(len(differences) - 1):
        if abs(differences[i] + differences[i + 1]) < tolerance and abs(differences[i]) > 0:
            counter += 1

    # Step 6: Sprawdzenie wartości progowej licznika
    if counter > threshold:
        return True  # Stego Image
    else:
        return False  # Normal Image


def analyze_images(cover_dir, stego_dir, threshold):
    cover_files = [os.path.join(cover_dir, f) for f in os.listdir(cover_dir) if os.path.isfile(os.path.join(cover_dir, f))][:20]
    stego_files = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir) if os.path.isfile(os.path.join(stego_dir, f))][:20]
    compress_files = [os.path.join(compress_dir, f) for f in os.listdir(compress_dir) if os.path.isfile(os.path.join(compress_dir, f))][:20]

    correct_stego = 0
    compress_stego = 0
    

    for cover_file, stego_file, compress_file in zip(cover_files, stego_files, compress_files):
        print(f"Analyzing Cover Image: {cover_file}")
        if not detect_stego_image(cover_file, compress_file, threshold):
            compress_stego += 1

        print(f"Analyzing Stego Image: {stego_file}")
        if detect_stego_image(cover_file, stego_file, threshold):
            correct_stego += 1

    print("\nAnalysis Results:")
    print(f"Correctly classified Cover Images: {compress_stego} / 20")
    print(f"Correctly classified Stego Images: {correct_stego} / 20")


cover_dir = "./bossbase_data/cover"
stego_dir = "./bossbase_data/stego"
compress_dir = "./bossbase_data/cover_compressed"


# Wartośc progowa detekcji
threshold_value = 4

analyze_images(cover_dir, stego_dir, threshold_value)
