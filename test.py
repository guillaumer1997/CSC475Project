import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
import PNMF as pnmf
from scipy import signal
from scipy.stats.mstats import gmean
from scipy.stats import pearsonr
from pydub import effects
import pydub

def rms_normalize(data):
    return effects.normalize(data)

DUMMY_SAMPLE = "/home/guillaume/Music/samples/sunshine(sample).wav"
SAMPLE_PATH = "/home/guillaume/Music/samples/OpenYourEyes(Sample).wav"
SONG_PATH = "/home/guillaume/Music/songs/the_light.wav"

song = rms_normalize(pydub.AudioSegment.from_wav(SONG_PATH))
#song_signal, fs = lb.load(SONG_PATH, sr = 22050)
#sample_signal, fs = lb.load(DUMMY_SAMPLE, sr = 22050)
dummy = rms_normalize(pydub.AudioSegment.from_wav(DUMMY_SAMPLE))
sample = rms_normalize(pydub.AudioSegment.from_wav(SAMPLE_PATH))

song_signal = np.array(song.get_array_of_samples())
sample_signal = np.array(sample.get_array_of_samples())
dummy_signal = np.array(dummy.get_array_of_samples())
def sinusoid(freq=440.0, dur=1.0, srate=44100.0, amp=1.0, phase = 0.0): 
    t = np.linspace(0,dur,int(srate*dur))
    data = amp * np.sin(2*np.pi*freq *t+phase)
    return data


def cross_corr(Ho, HD, k_components):
    acts = [np.array(signal.correlate(Ho[0], HD[0]))]
    print(acts)
    for x in range(1, k_components):
        acts = np.append(acts, [signal.correlate(Ho[x], HD[x])], axis = 0)
    return gmean(np.power(acts, 1/k_components), axis=0)


sample_fft = lb.stft(sample_signal.astype(float))
song_fft = lb.stft(song_signal.astype(float))
dummy_fft = lb.stft(dummy_signal.astype(float))
song_mag, song_phase = lb.magphase(song_fft)
sample_mag, sample_phase = lb.magphase(sample_fft)
dummy_mag,dummy_phase = lb.magphase(dummy_fft)

k_components = 10
l_components = 20

Wo, Ho = lb.decompose.decompose(np.abs(sample_mag), n_components = k_components, sort=True)
Wo_dum, Ho_dum = lb.decompose.decompose(np.abs(dummy_mag), n_components= k_components, sort=True)

HO = []
Hd = []
Hh = []
WD, HD, WH, HH = pnmf.PNMF(np.abs(song_mag), Wo,HO , Hd,Hh , l_components, 0)
WD_dum, HD_dum, WH_dum, HH_dum = pnmf.PNMF(np.abs(song_mag), Wo_dum, HO, Hd, Hh, l_components, 0)
k = cross_corr(Ho, HD, k_components)
plt.plot(k, c = 'red')

plt.plot(cross_corr(Ho_dum, HD_dum, k_components), c = 'blue')
plt.show()