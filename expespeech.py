import scipy.io as scipyio
import os
import librosa
import librosa.display as display
import pandas as pd

path = os.getcwd() + "/dataspeech/"

tv = pd.read_csv(path + "beautiful_TV.csv", header=None).values
a = pd.read_csv(path + "beautiful.csv", header=None).values

y, sr = librosa.load(path + 'beautiful.wav')
tab = pd.read_csv(path + "beautiful.csv", header=None)


testmfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)

librosa.display.specshow(testmfcc, sr=sr, x_axis='time')