import numpy as np
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
import PNMF as pnmf
from scipy import signal
import scipy
from scipy.stats.mstats import gmean
from pydub import effects
import pydub
from sklearn.metrics import pairwise_distances
import librosa.display
def rms_normalize(data):
    return effects.normalize(data)

def normalize(data):
    for i in range(0,len(data)):
        data[i] = data[i]/data.max()
    return data
    

DUMMY_SAMPLE = "/home/guillaume/Music/samples/OpenYourEyes(dummy).wav"
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
    acts = [np.array(signal.correlate(Ho[0], HD[0])/len(HD[0]))]
    print(acts)
    for x in range(1, k_components):
        acts = np.append(acts, [signal.correlate(Ho[x], HD[x], mode = 'full')/len(HD[x])], axis = 0)
    return np.power(gmean(acts, axis=0), 1/k_components)


sample_fft = lb.stft(sample_signal.astype(float))
song_fft = lb.stft(song_signal.astype(float))
dummy_fft = lb.stft(dummy_signal.astype(float))
song_mag, song_phase = lb.magphase(song_fft)
sample_mag, sample_phase = lb.magphase(sample_fft)
dummy_mag,dummy_phase = lb.magphase(dummy_fft)

k_components = 10
l_components = 20

Wo, Ho = lb.decompose.decompose(np.abs(sample_mag), n_components = k_components, sort=True)
#Wo_dum, Ho_dum = lb.decompose.decompose(np.abs(dummy_mag), n_components= k_components, sort=True)

HO = []
Hd = []
Hh = []
WD, HD, WH, HH = pnmf.PNMF(np.abs(song_mag), Wo,HO , Hd,Hh , l_components, 0)
HO = []
Hd = []
Hh = []
#WD_dum, HD_dum, WH_dum, HH_dum = pnmf.PNMF(np.abs(song_mag), Wo_dum, HO, Hd, Hh, l_components, 0)
#
#print(HD.max())
Ho = normalize(Ho)
HD = normalize(HD)
#HD_dum = normalize(HD_dum)
#Ho_dum = normalize(Ho_dum)
#print(HD_dum.max())
#print()
#k = cross_corr(Ho, HD, k_components)
#print(k.max())
#plt.plot(k[30000:40000], c = 'red')

#plt.plot(cross_corr(Ho_dum, HD_dum, k_components)[30000:40000], c = 'blue')
#plt.show()
#D, wp = lb.sequence.dtw(Ho[0], HD[0])

D = pairwise_distances(Ho.T, HD.T, 'correlation' )
#print(np.shape(D))


#step_sizes = np.zeros((20,2))
#for i,j in enumerate(range(0, int(20 *(len(HD[0]) - np.floor(len(Ho[1])/4))) , int((len(HD[1]) - np.floor(len(Ho[1])/4))))):
#    step_sizes[i,0] = i * (len(HD[0]) - np.floor(len(Ho[0])/4))
#    step_sizes[i,1] = (i + 1) * (len(HD[0]) - np.floor(len(Ho[0])/4))
#print(step_sizes)
#step_sizes = step_sizes.astype(int)




def dtw(x, y, table):
    i = len(x)
    j = len(y)
    path = [(i, j)]
    while i > 0 or j > 0:
        minval = np.inf
        if table[i-1][j-1] < minval:
            minval = table[i-1, j-1]
            step = (i-1, j-1)
        if table[i-1, j] < minval:
            minval = table[i-1, j]
            step = (i-1, j)
        if table[i][j-1] < minval:
            minval = table[i, j-1]
            step = (i, j-1)
        path.insert(0, step)
        i, j = step
    return np.array(path)






#path = dtw(Ho.T, HD.T, D)

C, wp = lb.sequence.dtw(C = D, subseq=True)

plt.subplot(2, 1, 1)
librosa.display.specshow(C, x_axis='frames', y_axis='frames')
#plt.plot(C[-1, :] / wp.shape[0])
plt.show()