import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
import PNMF as pnmf
from scipy import signal
from scipy.stats.mstats import gmean
from scipy.stats import pearsonr

def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.sum()/len(a))
DUMMY_SAMPLE = "/home/guillaume/Music/samples/oasis_live_forever(sample).wav"
SAMPLE_PATH = "/home/guillaume/Music/samples/OpenYourEyes(Sample).wav"
SONG_PATH = "/home/guillaume/Music/songs/the_light.wav"

song_signal, fs = lb.load(SONG_PATH, sr = 22050)
sample_signal, fs = lb.load(SAMPLE_PATH, sr = 22050)

sample_fft = lb.stft(sample_signal)
song_fft = lb.stft(song_signal)

song_mag, song_phase = lb.magphase(song_fft)
sample_mag, sample_phase = lb.magphase(sample_fft)

k_components = 10
l_components = 20

Wo, Ho = lb.decompose.decompose(np.abs(sample_mag), n_components = k_components, sort = True)

HO = []
Hd = []
Hh = []
WD, HD, WH, HH = pnmf.PNMF(np.abs(song_mag), Wo,HO , Hd,Hh , 20, 0)

cross1 = np.correlate(HD[0], Ho[0], "full")
cross2 = signal.correlate(HD[1], Ho[1], mode = 'full', method = 'fft')

plt.plot(cross1)
plt.show()