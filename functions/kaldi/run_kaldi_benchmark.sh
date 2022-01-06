
PATH=$PATH:

echo "[SPECTROGRAM]"
time ./kaldi/src/featbin/compute-spectrogram-feats scp:wav.scp ark:spectrogram.ark
time ./kaldi/src/featbin/compute-spectrogram-feats scp:wav.scp ark:spectrogram.ark
time ./kaldi/src/featbin/compute-spectrogram-feats scp:wav.scp ark:spectrogram.ark
time ./kaldi/src/featbin/compute-spectrogram-feats scp:wav.scp ark:spectrogram.ark
time ./kaldi/src/featbin/compute-spectrogram-feats scp:wav.scp ark:spectrogram.ark

echo "[MFCC]"
time ./kaldi/src/featbin/compute-mfcc-feats scp:wav.scp ark:mfcc.ark
time ./kaldi/src/featbin/compute-mfcc-feats scp:wav.scp ark:mfcc.ark
time ./kaldi/src/featbin/compute-mfcc-feats scp:wav.scp ark:mfcc.ark
time ./kaldi/src/featbin/compute-mfcc-feats scp:wav.scp ark:mfcc.ark
time ./kaldi/src/featbin/compute-mfcc-feats scp:wav.scp ark:mfcc.ark

echo "[FBANK]"
time ./kaldi/src/featbin/compute-fbank-feats scp:wav.scp ark:fbank.ark
time ./kaldi/src/featbin/compute-fbank-feats scp:wav.scp ark:fbank.ark
time ./kaldi/src/featbin/compute-fbank-feats scp:wav.scp ark:fbank.ark
time ./kaldi/src/featbin/compute-fbank-feats scp:wav.scp ark:fbank.ark
time ./kaldi/src/featbin/compute-fbank-feats scp:wav.scp ark:fbank.ark
