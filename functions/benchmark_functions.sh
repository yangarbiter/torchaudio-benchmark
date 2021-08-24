##!/usr/bin/env bash

mkdir -p results

python filters.py

python phase_vocoder.py

python griffinlim.py

python mfcc.py

python spectral_centroid.py

python spectrogram.py
