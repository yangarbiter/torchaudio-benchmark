

#### Installation

```bash
pip install numpy scipy espnet tqdm librosa
```

Install SoX
```bash
wget https://netactuate.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz

tar -xf sox-14.4.2.tar.gz
cd sox-14.4.2
./configure
make
```

To run benchmark for filters, you need to install sox and run
```bash
PATH=$HOME/torchaudio-benchmark/sox-14.4.2/src:$PATH python filters.py
```
