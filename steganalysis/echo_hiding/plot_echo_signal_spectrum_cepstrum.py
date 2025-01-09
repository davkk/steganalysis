import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

fs, signal1 = wavfile.read(sys.argv[1])
signal1 = signal1.astype(np.float64)

_, signal2 = wavfile.read(sys.argv[2])
signal2 = signal2.astype(np.float64)

time = np.linspace(0, 1, fs, endpoint=False)

N = signal1.size
freqs = np.fft.fftfreq(N, 1 / fs)

power_spec1 = np.abs(np.fft.fft(signal1)) ** 2
cepstrum1 = np.abs(np.fft.ifft(np.log(power_spec1 + 1e-10))) ** 2

power_spec2 = np.abs(np.fft.fft(signal2)) ** 2
cepstrum2 = np.abs(np.fft.ifft(np.log(power_spec2 + 1e-10))) ** 2

fig, axs = plt.subplots(nrows=3, ncols=1, tight_layout=True)
[ax_sign, ax_spec, ax_ceps] = axs

ax_sign.plot(signal1, label="bez echa")
ax_sign.plot(signal2, label="z echem")
ax_sign.set_title("Sygnał Audio")
ax_sign.set_xlabel("Czas [s]")
ax_sign.set_ylabel("Amplituda")
ax_sign.legend()

ax_spec.plot(freqs[:20000], power_spec1[:20000], label="bez echa")
ax_spec.plot(freqs[:20000], power_spec2[:20000], label="z echem")
ax_spec.set_title("Spektrum mocy sygnału")
ax_spec.set_xlabel("Częstotliwość [Hz]")
ax_spec.set_ylabel("Moc")
ax_spec.legend()

ax_ceps.plot(cepstrum1[10:1000], label="bez echa")
ax_ceps.plot(cepstrum2[10:1000], label="z echem")
ax_ceps.set_title("Cepstrum mocy")
ax_ceps.set_xlabel("Quefrency [próbki]")
ax_ceps.set_ylabel("Amplituda")
ax_ceps.legend()

fig.savefig(Path(__file__).with_suffix(".pdf"))
plt.show()
