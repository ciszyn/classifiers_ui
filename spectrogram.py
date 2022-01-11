import os
from kivy.clock import mainthread
import librosa
import numpy
import skimage.io
from pathlib import Path
from scipy.io import wavfile
import cv2


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X-X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, hop_length, n_mels):
    if y.size < hop_length:
        hop_length = y.size
    mels = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    mels = numpy.log(mels + 1e-9)
    img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0)
    img = 255-img
    return img


@mainthread
def create_spectrograms(filename, result, list, spinner):
    images = []
    try:
        x, sr = librosa.load(filename, offset=0.0, sr=None, mono=True)
        for i in range(int(x.size / sr / 3.)):
            lower = 3*i*sr
            if lower + 3*sr > x.size:
                upper = x.size
            else:
                upper = lower + 3*sr
            img = spectrogram_image(x[3*i*sr:upper], sr=sr,
                                    out="image"+str(i)+".png", hop_length=512, n_mels=128)
            images.append(img)
    except BaseException as e:
        print(e)
    result.img = images
    spinner.active = False

    # return images


def create_spectrograms_for_classif(filename):
    images = []
    try:
        x, sr = librosa.load(filename, offset=0.0, sr=None, mono=True)
        for i in range(int(x.size / sr / 3.)):
            lower = 3*i*sr
            if lower + 3*sr > x.size:
                upper = x.size
            else:
                upper = lower + 3*sr
            img = spectrogram_image(x[3*i*sr:upper], sr=sr,
                                    out="image"+str(i)+".png", hop_length=512, n_mels=128)
            images.append(img)
    except BaseException as e:
        print(e)

    return images
