import numpy as np
import pandas as pd
import pqkmeans
import pickle
import sys
import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import glob

FOLDER = sys.argv[1] # stimuli in .wav

features = {}
for filepath in glob.iglob(FOLDER + '/*/*.wav', recursive=True):
    wav_file = filepath.split('/')[-1]
    (rate,sig) = wav.read(filepath)
    mfcc_feat = mfcc(sig,rate)
    features[wav_file] = [mfcc_feat]

f = open("mfcc.pkl","wb")
pickle.dump(features, f)
f.close()