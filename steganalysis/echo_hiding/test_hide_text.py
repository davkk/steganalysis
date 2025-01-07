import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from steganalysis.echo_hiding.echo_hiding import EchoHiding, bits_to_text, text_to_bits

parser = argparse.ArgumentParser(description="hide image in wav file")
parser.add_argument("--audio", type=Path, required=True)
parser.add_argument("--message", type=str, required=True)
parser.add_argument("-A", type=float, default=1.0)
parser.add_argument("--offset", type=float, required=True)
parser.add_argument("--delta", type=float, default=5e-3)
parser.add_argument("-a", type=float, default=0.5)
args = parser.parse_args()

sample_rate, audio = wavfile.read(args.audio)
audio = audio.astype(np.floating)

eh = EchoHiding(
    amplitude=args.A,
    offset=int(args.offset * sample_rate),
    delta=int(args.delta * sample_rate),
    alpha=args.a,
)

watermark = np.array(list(map(int, text_to_bits(args.message))))

result = eh.encode(audio, sample_rate, bits=watermark)
wavfile.write(
    args.audio.with_stem(f"{args.audio.stem}-encoded"),
    sample_rate,
    result.astype(np.int16),
)

bits = eh.decode(result, num_bits=len(watermark))
decoded = bits_to_text("".join(str(bit) for bit in bits))

print(f"{args.message=}")
print(f"{decoded=}")
