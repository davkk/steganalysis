import sys
from typing import Any, Generator

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile


def power_cepstrum(audio: npt.NDArray[np.floating]):
    spectrum = np.abs(np.fft.fft(audio)) ** 2
    log_spectrum = np.log(spectrum + np.finfo(float).eps)
    power_cepstrum = np.abs(np.fft.ifft(log_spectrum)) ** 2
    return power_cepstrum


def iter_cepstrums(
    *,
    audio_size,
    window_size,
    step_size,
) -> Generator[npt.NDArray[np.int_], Any, Any]:
    for idx in range(0, (audio_size - window_size) + 1, step_size):
        window = audio[idx : idx + window_size]
        window = window * np.hamming(window_size)
        yield power_cepstrum(window)


class EchoDetector:
    def __init__(
        self,
        audio,
        sample_rate,
        *,
        window_size,
        step_size,
        min_delay_ms=1e-3,
        max_delay_ms=3e-2,
    ) -> None:
        self.audio = audio
        self.sample_rate = sample_rate

        self.min_delay = int(min_delay_ms * sample_rate)
        self.max_delay = int(max_delay_ms * sample_rate)

        self.window_size = window_size
        self.step_size = step_size

    def estimate_delays(self) -> tuple[int, int]:
        peak_locations = []

        for cepstrum in iter_cepstrums(
            audio_size=self.audio.size,
            window_size=self.window_size,
            step_size=self.step_size,
        ):
            peak_idx = np.argmax(cepstrum[self.min_delay : self.max_delay])
            peak_locations.append(self.min_delay + peak_idx)

        counts, bins = np.histogram(
            peak_locations,
            bins=(self.max_delay - self.min_delay) // 2,
            range=(self.min_delay, self.max_delay),
        )

        top_2 = np.argsort(counts)[-2:]
        d0, d1 = bins[top_2].astype(int)
        return d0, d1

    def estimate_segment_length(self, *, d0, d1):
        assert d0 > d1

        lengths = []
        sums = []

        est_length = self.audio.size // 2
        while est_length > d0:
            sum_a = []
            for cepstrum in iter_cepstrums(
                audio_size=self.audio.size,
                window_size=est_length,
                step_size=est_length,
            ):
                sum_a.append(cepstrum[d0] + cepstrum[d1])

            lengths.append(est_length)
            sums.append(np.mean(sum_a))
            est_length = int(est_length / 1.5)

        return lengths[np.argmax(sums)]


if __name__ == "__main__":
    sr, audio = wavfile.read(sys.argv[1])
    detector = EchoDetector(
        audio,
        sr,
        window_size=2048,
        step_size=16,
    )

    d0, d1 = detector.estimate_delays()
    print(d0, d1)

    segment_len = detector.estimate_segment_length(d0=d0, d1=d1)
    print(segment_len)
