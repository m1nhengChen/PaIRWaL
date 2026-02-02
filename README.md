# PaIRWaL
Probability-Invariant Random-Walk Learning framework for gyral folding networks classification.

Code for the data preprocessing part can be found in ./data_procssing, including the generation of gyral folding network-based cortical similarity and the extraction of 3HG fingerprint.

Run the following command to train the model:
```
python run_schemeA.py \
  --in-npy tcsva_5ring_3hg.npy \
  --out-dir ./rwnn \
  --fp-dir /mnt/raid/MIND/FINGERPRINT_3hg_5ring \
  --sparsify density --density 0.10 --keep-mst \
  --mode multiclass --classes 0,2,3 --kfold 5 \
  --walk-len 64 --non-backtracking --named-neighbors \
  --n-walks-train 8 --n-walks-eval 64 \
  --reader gru --emb-dim 128 --hidden 128

```
