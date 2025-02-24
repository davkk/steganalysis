from typing import Any, Generator

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt


def power_cepstrum(audio: npt.NDArray[np.floating]):
    spectrum = np.abs(np.fft.fft(audio)) ** 2
    log_spectrum = np.log(spectrum + np.finfo(float).eps)
    power_cepstrum = np.abs(np.fft.ifft(log_spectrum)) ** 2
    return power_cepstrum


class EchoDetector:
    def __init__(
        self,
        window_size,
        step_size,
        min_delay,
        max_delay,
    ) -> None:
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.window_size = window_size
        self.step_size = step_size

    def iter_cepstrums(
        self,
        audio,
        *,
        window_size=None,
        step_size=None,
    ) -> Generator[npt.NDArray[np.int_], Any, Any]:
        window_size = window_size or self.window_size
        step_size = step_size or self.step_size
        for idx in range(0, (audio.size - window_size) + 1, step_size):
            window = audio[idx : idx + window_size]
            window = window * np.hamming(window_size)
            yield power_cepstrum(window)

    def estimate_delays(
        self,
        audio: npt.NDArray[np.floating],
        *,
        save_plot: bool = False,
    ) -> tuple[float, int, int]:
        peak_locations = [
            self.min_delay + np.argmax(cepstrum[self.min_delay : self.max_delay])
            for cepstrum in self.iter_cepstrums(audio)
        ]

        counts, bins = np.histogram(
            peak_locations,
            bins=(self.max_delay - self.min_delay),
            range=(self.min_delay, self.max_delay),
        )

        top_2 = np.argsort(counts)[-2:]
        d0, d1 = bins[top_2]

        if save_plot:
            plt.figure(figsize=(9, 4))
            plt.bar(bins[:-1], counts, width=(bins[1] - bins[0]))
            plt.axvline(
                x=d0,
                linestyle="-.",
                color="orange",
                alpha=0.6,
                label=f"$d_0 = {d0:.2f}$ [próbki]",
            )
            plt.axvline(
                x=d1,
                linestyle="--",
                color="orange",
                alpha=0.6,
                label=f"$d_1 = {d1:.2f}$ [próbki]",
            )
            plt.title("Histogram lokalizacji maksymalnych wartości")
            plt.xlabel("Opóźnienie echa [próbki]")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig("plot_echo_peak_location.pdf")

        cplar = sum(counts[top_2]) / len(peak_locations)
        return cplar, int(d0), int(d1)

    def estimate_segment_length(
        self,
        audio: npt.NDArray[np.floating],
        sample_rate: int,
        *,
        d0: int,
        d1: int,
        save_plot: bool = False,
    ):
        assert d0 > d1

        lengths = []
        avg_sums = []

        step = int(1e-3 * sample_rate)
        for est_length in range(2 * d0, audio.size // (8 * 8), step):
            avg_sum = np.mean(
                [
                    cepstrum[d0] + cepstrum[d1]
                    for cepstrum in self.iter_cepstrums(
                        audio,
                        window_size=est_length,
                        step_size=est_length,
                    )
                ]
            )

            lengths.append(est_length)
            avg_sums.append(avg_sum)

        if save_plot:
            plt.figure(figsize=(6, 4))
            plt.plot(lengths, avg_sums, ".-")
            plt.title("Estymacja długości segmentu")
            plt.xlabel("Długość segmentu [próbki]")
            plt.ylabel("Średnia suma wartości mocy cepstrum w $d_0$ i $d_1$")
            plt.tight_layout()
            plt.grid(True)
            plt.savefig("plot_echo_segment_length.pdf")

        return lengths[np.argmax(avg_sums)]
