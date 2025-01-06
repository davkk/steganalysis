import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def power_cepstrum(signal):
    spectrum = np.abs(np.fft.fft(signal)) ** 2
    log_spectrum = np.log(spectrum + np.finfo(float).eps)
    power_cepstrum = np.abs(np.fft.ifft(log_spectrum)) ** 2
    return power_cepstrum


def estimate_delays(
    audio,
    sample_rate,
    *,
    window_size,
    step_size,
    min_delay_ms=1e-3,
    max_delay_ms=4e-2,
):
    min_delay = int(min_delay_ms * sample_rate)
    max_delay = int(max_delay_ms * sample_rate)

    peak_locations = []

    for idx in range(0, (audio.size - window_size) + 1, step_size):
        window = audio[idx : idx + window_size]
        window = window * np.hamming(window.size)
        cepstrum = power_cepstrum(window)

        peak_idx = np.argmax(cepstrum[min_delay:max_delay])
        peak_locations.append(min_delay + peak_idx)

    counts, bins = np.histogram(
        peak_locations,
        bins=(max_delay - min_delay) // 2,
        range=(min_delay, max_delay),
    )

    [d0, d1, *_] = map(int, bins[np.argsort(counts)[-2:][::-1]])

    # plt.bar(bins[:-1], counts)
    # plt.xlabel("Delay (samples)")
    # plt.ylabel("Frequency")
    # plt.title("Echo Delay Histogram")
    # plt.show()

    lengths = []
    sums = []

    est_length = audio.size
    while est_length > d1:
        sum_a = 0
        for idx in range(0, audio.size - est_length + 1, est_length):
            window = audio[idx : idx + est_length]
            window = window * np.hamming(window.size)
            cepstrum = power_cepstrum(window)
            if d0 < cepstrum.size and d1 < cepstrum.size:
                sum_a += cepstrum[d0] + cepstrum[d1]
        lengths.append(est_length)
        sums.append(sum_a)
        est_length //= 2

    return (d0, d1), lengths[np.argmax(sums)]


if __name__ == "__main__":
    sr, audio = wavfile.read(sys.argv[1])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # convert to mono

    print(
        estimate_delays(
            audio,
            sr,
            window_size=512,
            step_size=8,
        )
    )
