import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from steganalysis.echo_hiding.echo_detector import EchoDetector
from steganalysis.echo_hiding.echo_hiding import EchoHiding, bits_to_text, text_to_bits

parser = argparse.ArgumentParser(description="hide image in wav file")
parser.add_argument("--audio", type=Path, required=True)
parser.add_argument("--message", type=str, required=True)
parser.add_argument("-A", type=float, default=1.0)
parser.add_argument("-a", type=float, default=0)
parser.add_argument("--offset", type=int, required=True)
parser.add_argument("--delta", type=int, required=True)
parser.add_argument("--window-size", type=int, required=True)
parser.add_argument("--step-size", type=int, required=True)
parser.add_argument("--save-plot", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

sample_rate, audio = wavfile.read(args.audio)
audio = audio.astype(np.float64)

eh = EchoHiding(
    amplitude=args.A,
    offset=args.offset,
    delta=args.delta,
    alpha=args.a,
)

watermark = np.array(list(map(int, text_to_bits(args.message))))

result = eh.encode(audio, sample_rate, bits=watermark)
wavfile.write(
    args.audio.with_stem(f"{args.audio.stem}-encoded"),
    sample_rate,
    result.astype(np.int16),
)

detector = EchoDetector(
    window_size=args.window_size,
    step_size=args.step_size,
    min_delay=int(1e-3 * sample_rate),
    max_delay=int(3e-2 * sample_rate),
)

CPLAR, d0, d1 = detector.estimate_delays(
    result,
    save_plot=args.save_plot,
)
print(f"{CPLAR=:.3f}")
assert CPLAR > 0.5

print(f"offset = {d1}")
print(f"delta = {d0 - d1}")

segment_len = detector.estimate_segment_length(
    result,
    sample_rate,
    d0=d0,
    d1=d1,
    save_plot=args.save_plot,
)
print(f"{segment_len=}")

num_bits = result.size // segment_len
bits = np.zeros(num_bits).astype(np.uint8)

for idx, cepstrum in zip(
    range(num_bits),
    detector.iter_cepstrums(
        result,
        window_size=segment_len,
        step_size=segment_len,
    ),
):
    peak_one = d1 < cepstrum.size and cepstrum[d1] or 0
    peak_zero = d0 < cepstrum.size and cepstrum[d0] or 0
    bits[idx] = int(peak_one < peak_zero)

decoded = bits_to_text("".join(str(bit) for bit in bits))

print(f"{args.message=}")
print(f"{decoded=}")
