import numpy as np
import numpy.typing as npt
import scipy.fftpack as fft
from scipy.signal import fftconvolve


def text_to_bits(text: str) -> str:
    return "".join(f"{ord(char):08b}" for char in text)


def bits_to_text(bits: str) -> str:
    return "".join(chr(int(bits[i : i + 8], 2)) for i in range(0, len(bits), 8))


class EchoHiding:
    def __init__(self, *, amplitude: float, offset: float, delta: float, alpha: float):
        self.amplitude = amplitude
        self.offset = offset
        self.delta = delta
        self.alpha = alpha
        print(self.offset, self.offset + self.delta)

    def encode(
        self,
        signal: npt.NDArray[np.int_],
        *,
        bits: npt.NDArray[np.int_],
        sample_rate: int,
    ) -> npt.NDArray[np.int_]:
        segment_len = len(signal) // len(bits)
        print(f"{signal.size=}")
        print(f"{segment_len=}")
        result = np.zeros_like(signal)

        for i, bit in enumerate(bits):
            delay = self.offset + (bit * self.delta)

            kernel = np.zeros(delay + 1)
            assert (
                kernel.size < segment_len
            ), f"kernel should fit in segment ({kernel.size} >= {segment_len})"

            kernel[0] = 1.0
            kernel[delay] = self.amplitude * np.exp(-self.alpha * delay / sample_rate)

            start = i * segment_len
            end = start + segment_len
            result[start:end] += fftconvolve(signal[start:end], kernel, mode="same")

        return result

    def decode(
        self,
        signal: npt.NDArray[np.int_],
        *,
        num_bits: int,
    ) -> npt.NDArray[np.uint8]:
        bits = np.zeros(num_bits).astype(np.uint8)
        segment_len = len(signal) // num_bits

        for i in range(num_bits):
            start = i * segment_len
            end = start + segment_len

            segment = signal[start:end]
            cepstrum = np.abs(fft.ifft(np.log(np.abs(fft.fft(segment)) ** 2 + 1e-10)))
            assert self.offset + self.delta < len(cepstrum)

            peak_zero = cepstrum[self.offset]
            peak_one = cepstrum[self.offset + self.delta]

            bits[i] = int(peak_one > peak_zero)

        return bits
