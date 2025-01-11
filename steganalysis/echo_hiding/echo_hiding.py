import numpy as np
import numpy.typing as npt


def text_to_bits(text: str) -> str:
    return "".join(f"{ord(char):08b}" for char in text)


def bits_to_text(bits: str) -> str:
    return "".join(chr(int(bits[i : i + 8], 2)) for i in range(0, len(bits), 8))


class EchoHiding:
    def __init__(self, *, amplitude: float, offset: int, delta: int, alpha: float):
        self.amplitude = amplitude
        self.offset = offset
        self.delta = delta
        self.alpha = alpha

    def encode(
        self,
        signal: npt.NDArray[np.floating],
        sample_rate: int,
        *,
        bits: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        segment_len = signal.size // bits.size
        print(f"real {segment_len=}")
        result = np.zeros_like(signal)

        for idx, bit in enumerate(bits):
            delay = self.offset + (bit * self.delta)

            kernel = np.zeros(delay + 1).astype(np.floating)
            assert (
                kernel.size < segment_len
            ), f"kernel should fit in segment ({kernel.size} >= {segment_len})"

            kernel[0] = 1.0
            kernel[delay] = self.amplitude * np.exp(-self.alpha * delay / sample_rate)

            start = idx * segment_len
            end = start + segment_len
            window = signal[start:end] * np.hamming(segment_len)
            result[start:end] += np.convolve(window, kernel, mode="same")

        return result
