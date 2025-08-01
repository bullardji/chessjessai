# ChessJessAI

This repository contains an experimental chess AI prototype based on a King-relative NNUE encoder and a JEPA world model.

## C acceleration

A small C library `libfast_linear.so` accelerates the first linear layer of the encoder. Build it with:

```sh
gcc -O3 -march=native -fPIC -shared fast_linear.c -o libfast_linear.so
```

If the shared library is present, the Python module will use it automatically on CPU.

### Weight clipping

The NNUE encoder exposes `clip_weights()` which clamps the first-layer
parameters to the Stockfish range (±127/64). Call this periodically during
training to keep the network compatible with int8 quantisation.

## Dependencies

Install the required Python packages:

```sh
pip install torch chess zstandard tqdm
```

## Stage 0 training data

Stage 0 uses the public Lichess evaluation dataset `lichess_db_eval.jsonl.zst`.
Download it with:

```sh
wget https://database.lichess.org/lichess_db_eval.jsonl.zst
```

Then run the training script:

```sh
python training.py --data lichess_db_eval.jsonl.zst --epochs 1 --batch 32
```

The `training.py` loader streams the compressed file directly using the
`zstandard` module, so it never needs to be fully decompressed on disk. A
`tqdm` progress bar reports the running loss and estimated time each epoch.

## Colab training

A ready-to-run Colab notebook `colab_training.ipynb` automates the setup and downloads.
Open it in Google Colab and execute the cells to compile the optional C library, fetch the dataset and start training on a GPU.
