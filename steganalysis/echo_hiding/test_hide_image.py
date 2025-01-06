import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.io import wavfile

from steganalysis.echo_hiding.echo_hiding import EchoHiding

parser = argparse.ArgumentParser(description="hide image in wav file")
parser.add_argument("--audio", type=Path, required=True)
parser.add_argument("--image", type=Path, required=True)
parser.add_argument("--size", type=int, required=True)
parser.add_argument("-A", type=float, default=1.0)
parser.add_argument("--offset", type=float, required=True)
parser.add_argument("--delta", type=float, default=5e-3)
parser.add_argument("-a", type=float, default=0.5)
args = parser.parse_args()

sample_rate, original = wavfile.read(args.audio)
if original.ndim > 1:
    original = original.mean(axis=1)  # convert to mono

eh = EchoHiding(
    amplitude=args.A,
    offset=int(args.offset * sample_rate),
    delta=int(args.delta * sample_rate),
    alpha=args.a,
)

image = Image.open(args.image)
width, height = image.size
target_size = min(width, height)
left = (width - target_size) // 2
top = (height - target_size) // 2
right = left + target_size
bottom = top + target_size
image = image.crop((left, top, right, bottom))

bw_image = image.resize((args.size, args.size), Image.Resampling.NEAREST).convert("1")
watermark = np.array(bw_image).flatten()

result = eh.encode(original, bits=watermark, sample_rate=sample_rate)
wavfile.write(
    args.audio.with_stem(f"encoded-{args.audio.stem}-{args.image.stem}"),
    sample_rate,
    result.astype(np.int16),
)

bits = eh.decode(result, num_bits=len(watermark))
image_bits = bits.reshape((args.size, args.size)).astype(np.uint8)
decoded_image = Image.fromarray(image_bits)

fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
[ax_or, ax_px, ax_de] = axs

ax_or.imshow(image)
ax_or.set_title("Original Image")
ax_or.axis("off")

ax_px.imshow(bw_image, cmap="gray")
ax_px.set_title("Pixelated Image")
ax_px.axis("off")

ax_de.imshow(decoded_image, cmap="gray")
ax_de.set_title("Decoded Image")
ax_de.axis("off")

fig.tight_layout()
plt.show()
