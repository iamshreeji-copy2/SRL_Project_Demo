import numpy as np
import librosa
from spafe.features.lfcc import lfcc
from spafe.utils.preprocessing import SlidingWindow

def cqtspec(data, sr, min_freq=30, octave_resolution=14):
    max_frequency = sr / 2
    num_freq = round(octave_resolution * np.log2(max_frequency / min_freq))
    cqt_spectrogram = np.abs(librosa.cqt(data, sr=sr, fmin=min_freq, bins_per_octave=octave_resolution, n_bins=num_freq))
    return cqt_spectrogram

def cqhc(data, sr, min_freq=30, octave_resolution=14, num_coeff=20):
    cqt_spectrogram = np.power(cqtspec(data, sr, min_freq, octave_resolution), 2)
    num_freq = np.shape(cqt_spectrogram)[0] 
    ftcqt_spectrogram = np.fft.fft(cqt_spectrogram, 2 * num_freq - 1, axis=0)
    absftcqt_spectrogram = abs(ftcqt_spectrogram)
    pitch_component = np.real(np.fft.ifft(ftcqt_spectrogram / (absftcqt_spectrogram + 1e-14), axis=0)[0:num_freq, :])
    coeff_indices = np.round(octave_resolution * np.log2(np.arange(1, num_coeff + 1))).astype(int)
    audio_cqhc = pitch_component[coeff_indices, :]
    return audio_cqhc

def lfcc_mine(sr, audio):
    lfccs = lfcc(audio, fs=sr, num_ceps=20, nfft=512, window=SlidingWindow(win_len=0.025, win_hop=0.010))
    return lfccs
