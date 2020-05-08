from shennong.audio import Audio
from shennong.features.processor.bottleneck import BottleneckProcessor
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import glob

# Path("./output_csvs").mkdir(parents=True, exist_ok=True)

FOLDER = sys.argv[1] # stimuli in .wav


all_features = dict()
for filepath in glob.iglob(FOLDER + '/*/*.wav', recursive=True):
    audio = Audio.load(filepath)
    wav_file = filepath.split('/')[-1]
    all_features[wav_file] = audio
    processor = BottleneckProcessor(weights='BabelMulti')
    features = processor.process_all(all_features)

d = dict()
for key in features:
    # access every features object
    feats = features[key].data
    d[key] = [feats]

df = pd.DataFrame.from_dict(d, orient='index')
df.to_csv('./out_feats.csv')