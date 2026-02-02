# PaIRWaL
Probability-Invariant Random-Walk Learning framework for gyral folding networks classification.
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
