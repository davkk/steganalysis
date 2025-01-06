import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from steganalysis.echo_hiding.echo_hiding import EchoHiding, bits_to_text, text_to_bits

file_path = Path(sys.argv[1])

sr, original = wavfile.read(file_path)
if original.ndim > 1:
    original = original.mean(axis=1)  # convert to mono

eh = EchoHiding(
    amplitude=1,
    offset=int(1e-2 * sr),
    delta=int(1e-2 * sr),
    alpha=0.5,
)

message = sys.argv[2]
watermark = np.array(list(map(int, text_to_bits(message))))

result = eh.encode(original, bits=watermark)
wavfile.write(
    file_path.with_stem(f"encoded-{file_path.stem}"),
    sr,
    result.astype(np.int16),
)

bits = eh.decode(result, num_bits=len(watermark))
decoded = bits_to_text("".join(str(bit) for bit in bits))

print(f"{message=}")
print(f"{decoded=}")
