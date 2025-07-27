import os
import io
import json
from typing import Iterator, Tuple
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset, DataLoader

import chess

from chess_ai_pure_python import (
    ChessAgent,
    move_to_index,
)

try:
    import zstandard as zstd
except ImportError:
    zstd = None

LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
LICHESS_EVAL_COUNT = 259_736_183  # total positions in the dataset

def download_lichess_eval(path: str) -> None:
    """Download the Lichess evaluation dataset if not present."""
    if os.path.exists(path):
        return
    import urllib.request
    print(f"Downloading {LICHESS_EVAL_URL} ...")
    urllib.request.urlretrieve(LICHESS_EVAL_URL, path)

class LichessEvalDataset(IterableDataset):
    """Stream evaluations from the lichess JSONL.zst file.

    The JSON lines file is kept compressed on disk and decompressed on the fly
    using a streaming reader, so memory usage stays low even for hundreds of
    millions of positions.
    """

    def __init__(self, zst_path: str):
        if zstd is None:
            raise ImportError("zstandard package is required: pip install zstandard")
        self.zst_path = zst_path

    def __iter__(self) -> Iterator[Tuple[chess.Board, int, float]]:
        with open(self.zst_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text:
                    obj = json.loads(line)
                    fen = obj["fen"]
                    board = chess.Board(fen)
                    evals = obj.get("evals", [])
                    if not evals:
                        continue
                    best = max(evals, key=lambda e: e.get("depth", 0))
                    pv = best.get("pvs", [])
                    if not pv:
                        continue
                    pv0 = pv[0]
                    line_moves = pv0.get("line", "").split()
                    if not line_moves:
                        continue
                    move = chess.Move.from_uci(line_moves[0])
                    cp = pv0.get("cp")
                    mate = pv0.get("mate")
                    if cp is None:
                        if mate is None:
                            continue
                        cp = 10000 * (1 if mate > 0 else -1)
                    move_idx = move_to_index(move)
                    yield board, move_idx, float(cp) / 100.0

def collate_batch(batch):
    boards, move_ids, values = zip(*batch)
    return list(boards), torch.tensor(move_ids, dtype=torch.long), torch.tensor(values, dtype=torch.float32)

def train_stage0(
    dataset_path: str,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cpu",
) -> ChessAgent:
    """Train the encoder and policy head on the Lichess dataset.

    The dataset is read as a compressed stream, so passing a `.zst` file is
    preferred and keeps disk usage low.
    """

    download_lichess_eval(dataset_path)
    agent = ChessAgent(device=device)
    dataset = LichessEvalDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
    params = list(agent.encoder.parameters()) + list(agent.world.parameters()) + list(agent.predictor.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    total_batches = LICHESS_EVAL_COUNT // batch_size
    for epoch in range(epochs):
        avg_loss = None
        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for boards, move_ids, values in loader:
                latents, values_pred = agent.encode_batch(boards)
                logits, val_out, _ = agent.predictor(latents)
                loss_policy = torch.nn.functional.cross_entropy(logits, move_ids.to(device))
                loss_value = torch.nn.functional.mse_loss(val_out, values.to(device))
                loss = loss_policy + loss_value
                opt.zero_grad()
                loss.backward()
                opt.step()
                if avg_loss is None:
                    avg_loss = loss.item()
                else:
                    avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
        print(f"epoch {epoch}: loss={avg_loss:.4f}")
    return agent

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train stage 0 on lichess eval data")
    parser.add_argument("--data", default="lichess_db_eval.jsonl.zst")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_stage0(args.data, args.epochs, args.batch, device=args.device)

