import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

fs, signal = wavfile.read(sys.argv[1])
signal = signal.astype(np.float64)
time = np.linspace(0, 1, fs, endpoint=False)

N = signal.size
freqs = np.fft.fftfreq(N, 1 / fs)
fft_values = np.fft.fft(signal)
power_spectrum = np.abs(fft_values) ** 2

log_power_spectrum = np.log(power_spectrum + 1e-10)
cepstrum = np.abs(np.fft.ifft(log_power_spectrum)) ** 2

fig, axs = plt.subplots(nrows=3, ncols=1, tight_layout=True)
[ax_sign, ax_spec, ax_ceps] = axs

ax_sign.plot(signal)
ax_sign.set_title("Sygnał Audio")
ax_sign.set_xlabel("Czas [s]")
ax_sign.set_ylabel("Amplituda")

ax_spec.plot(freqs[:20000], power_spectrum[:20000])
ax_spec.set_title("Spektrum mocy sygnału")
ax_spec.set_xlabel("Częstotliwość [Hz]")
ax_spec.set_ylabel("Moc")

ax_ceps.plot(cepstrum[10:1000])
ax_ceps.set_title("Cepstrum mocy")
ax_ceps.set_xlabel("Quefrency [próbek]")
ax_ceps.set_ylabel("Amplituda")

fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
