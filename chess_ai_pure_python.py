# chess_ai_advanced.py
# ===============================================================
# A fully‑working, research‑grade chess AI framework that merges
# the speed of king‑relative NNUE evaluation with a momentum‑target
# Joint‑Embedding Predictive Architecture (JEPA) world model and
# an MCTS‑ready multi‑head policy / value / ponder predictor.
#
# Key fixes & extensions compared with the previous draft:
#   • King‑relative incremental encoder with phase + material bits
#   • 20480‑way move indexing (from‑sq, to‑sq, promotion) + legal mask
#   • Momentum‑target JEPA training (BYOL‑style cosine loss)
#   • EMA target‑network update utility
#   • Predictor now masks illegal moves before soft‑max
#   • Cosine‑similarity model loss (stable scale)
#   • Clean separation of utility functions for move indexing / masks
#   • End‑to‑end demo (CPU/GPU) with minimal external deps
# ===============================================================

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import ctypes
import os

# ───────────────────────── C acceleration (optional) ────────────
_lib_path = os.path.join(os.path.dirname(__file__), "libfast_linear.so")
try:
    _fastlib = ctypes.CDLL(_lib_path)
    _fastlib.fc1_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # weights
        ctypes.POINTER(ctypes.c_float),  # bias
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_size_t,  # out_dim
        ctypes.c_size_t,  # in_dim
    ]
    USE_FAST_FC1 = True
except OSError:  # pragma: no cover - library missing
    _fastlib = None
    USE_FAST_FC1 = False

# ───────────────────────── 3rd‑party deps ─────────────────────────
try:
    import chess
    import chess.pgn
    import chess.polyglot
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "python‑chess is required for this module.  Install with `pip install chess`."
    ) from e

# ───────────────────────── Constants ─────────────────────────────
MOVE_PROMO_TO_ID = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
ID_TO_PROMO = {v: k for k, v in MOVE_PROMO_TO_ID.items()}

MOVE_VOCAB_SIZE = 5 * 64 * 64  # 20 480
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
STARTING_PHASE = 24  # Stockfish convention

# ───────────────────────── Utility functions ─────────────────────


def move_to_index(move: chess.Move) -> int:
    """Map a python‑chess Move to a unique integer in [0, 20479]."""
    promo_id = MOVE_PROMO_TO_ID.get(move.promotion, 0)
    return promo_id * 4096 + move.from_square * 64 + move.to_square


def index_to_move(index: int) -> chess.Move:
    """Inverse of `move_to_index` (for completeness)."""
    promo_id, rem = divmod(index, 4096)
    from_sq, to_sq = divmod(rem, 64)
    promo_piece = ID_TO_PROMO[promo_id]
    return chess.Move(from_sq, to_sq, promotion=promo_piece)


def legal_moves_mask(board: chess.Board, device: torch.device | None = None) -> torch.Tensor:
    """Return a bool mask of shape (MOVE_VOCAB_SIZE,) with True for legal moves."""
    mask = torch.zeros(MOVE_VOCAB_SIZE, dtype=torch.bool, device=device)
    for mv in board.legal_moves:
        mask[move_to_index(mv)] = True
    return mask


# ───────────────────────── King‑relative NNUE encoder ────────────
class KingRelativeNNUE(nn.Module):
    """
    Incrementally updatable NNUE‑style encoder with:
      • king‑centred, side‑to‑move perspective
      • 768 piece‑square bits (6 piece types × 64 squares × {stm, otm})
      • 1 side‑to‑move bit (kept for symmetry with old nets)
      • 4 castling bits
      • 8 en‑passant file bits
      • 3 half‑move clock buckets
      • 8 phase buckets
      • 16 material‑count features (piece counts for both colours)
    Total input = 808.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = 808
        self.fc1 = nn.Linear(self.input_dim, latent_dim)
        self.relu = nn.ReLU()
        self.value_head = nn.Linear(latent_dim, 1)

        # Range used when clamping weights for integer quantisation
        self.weight_clip = 127.0 / 64.0

        # Incremental cache {zobrist_hash(board) : latent tensor}
        self._cache: Dict[int, torch.Tensor] = {}

    # ─── Feature helpers ─────────────────────────────────────────

    @staticmethod
    def _phase_bucket(board: chess.Board) -> torch.Tensor:
        """Return one‑hot 8‑dim phase bucket."""
        remaining_phase = STARTING_PHASE
        piece_phase = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
        for piece_type in piece_phase:
            remaining_phase -= piece_phase[piece_type] * (
                len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
            )
        phase_idx = max(0, min(7, remaining_phase))
        one_hot = torch.zeros(8)
        one_hot[phase_idx] = 1.0
        return one_hot

    @staticmethod
    def _material_vector(board: chess.Board) -> torch.Tensor:
        """16‑dim material counts (STM pieces first 8, OTM next 8; king excluded)."""
        vec = torch.zeros(16)
        stm = board.turn
        others = not stm
        # Order: P, N, B, R, Q,  ‑ (unused), ‑, ‑   ×2
        order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        for i, pt in enumerate(order):
            vec[i] = len(board.pieces(pt, stm))
            vec[8 + i] = len(board.pieces(pt, others))
        return vec

    # ─── Board -> sparse feature vector ──────────────────────────
    def encode_board(self, board: chess.Board, device: torch.device | None = None) -> torch.Tensor:
        feats = torch.zeros(self.input_dim, dtype=torch.float32, device=device)

        # 1) Piece‑square features relative to STM king
        king_sq = board.king(board.turn)
        if king_sq is None:
            raise ValueError("Illegal board (no king).")
        stm = board.turn
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
            rel_sq = (square - king_sq) & 63
            type_idx = piece.piece_type - 1
            if piece.color != stm:
                type_idx += 6  # opponent pieces
            feats[12 * rel_sq + type_idx] = 1.0

        offset = 768

        # 2) Side‑to‑move bit (still useful for nets trained on symmetric data)
        feats[offset] = float(board.turn)
        offset += 1

        # 3) Castling rights (Wk, Wq, Bk, Bq)
        castling = [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ]
        feats[offset : offset + 4] = torch.tensor(castling, dtype=torch.float32, device=device)
        offset += 4

        # 4) En‑passant file
        if board.ep_square is not None:
            feats[offset + chess.square_file(board.ep_square)] = 1.0
        offset += 8

        # 5) Half‑move clock bucket (0‑25, 26‑50, 51‑75, ≥76)
        halfmove = board.halfmove_clock
        bucket = min(3, halfmove // 25)
        feats[offset + bucket] = 1.0
        offset += 3

        # 6) Phase bucket
        feats[offset : offset + 8] = self._phase_bucket(board).to(device)
        offset += 8

        # 7) Material vector
        feats[offset : offset + 16] = self._material_vector(board).to(device)

        return feats

    # ─── Forward (latent, value) ────────────────────────────────
    def forward(self, board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor]:
        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self._cache:
            latent = self._cache[board_hash]
        else:
            feats = self.encode_board(board, device=next(self.parameters()).device)
            if USE_FAST_FC1 and feats.device.type == "cpu":
                w = self.fc1.weight.detach().contiguous().to(torch.float32)
                b = self.fc1.bias.detach().contiguous().to(torch.float32)
                out = torch.empty(self.latent_dim, dtype=torch.float32)
                _fastlib.fc1_forward(
                    w.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    b.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    feats.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    self.latent_dim,
                    self.input_dim,
                )
                latent = self.relu(out)
            else:
                latent = self.relu(self.fc1(feats))
            self._cache[board_hash] = latent
        value = self.value_head(latent).squeeze(-1)
        return latent, value

    # ─── Optional dynamic quantisation ───────────────────────────
    def quantize(self) -> "KingRelativeNNUE":
        return torch.quantization.quantize_dynamic(self, {nn.Linear}, dtype=torch.qint8)  # type: ignore

    def clip_weights(self) -> None:
        """Clamp fc1 parameters to the recommended int8 range."""
        with torch.no_grad():
            self.fc1.weight.clamp_(-self.weight_clip, self.weight_clip)
            self.fc1.bias.clamp_(-self.weight_clip, self.weight_clip)


# ───────────────────── Block‑causal attention (pure PyTorch) ──────────────────
class BlockCausalAttention(nn.Module):
    """Simple block‑causal attention (no Flash kernels, runs everywhere)."""

    def __init__(self, embed_dim: int, num_heads: int, block_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        mask = torch.zeros(seq, seq, dtype=torch.bool, device=x.device)
        n_blocks = math.ceil(seq / self.block_size)
        for i in range(n_blocks):
            s_i, e_i = i * self.block_size, min((i + 1) * self.block_size, seq)
            for j in range(i + 1, n_blocks):
                s_j, e_j = j * self.block_size, min((j + 1) * self.block_size, seq)
                mask[s_i:e_i, s_j:e_j] = True
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


# ───────────────────── JEPA world model with target net ───────────────────────
class JepaWorldModel(nn.Module):
    """
    Joint‑Embedding Predictive Architecture (online stack).
    Move embedding uses from/to/promo decomposition and sums their vectors.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 512,
        block_size: int = 8,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.positional = nn.Parameter(torch.zeros(1, max_seq_len, latent_dim))
        # Move component embeddings
        self.embed_from = nn.Embedding(64, latent_dim)
        self.embed_to = nn.Embedding(64, latent_dim)
        self.embed_promo = nn.Embedding(5, latent_dim)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        BlockCausalAttention(latent_dim, num_heads, block_size),
                        nn.Linear(latent_dim, ff_dim),
                        nn.ReLU(),
                        nn.Linear(ff_dim, latent_dim),
                    ]
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(latent_dim)

    # ─── Move embedding helper ────────────────────────────────
    def _embed_move_ids(self, move_ids: torch.Tensor) -> torch.Tensor:
        # move_ids: (B, T)
        promo = move_ids // 4096
        rem = move_ids % 4096
        frm = rem // 64
        to = rem % 64
        return self.embed_from(frm) + self.embed_to(to) + self.embed_promo(promo)

    # ─── Forward ──────────────────────────────────────────────
    def forward(
        self,
        latents: torch.Tensor,  # (B, T, D)
        move_ids: Optional[torch.Tensor],  # (B, T)
        mask: torch.Tensor,  # (B, T) True = predict
    ) -> torch.Tensor:
        b, t, _ = latents.shape
        x = latents.clone()
        x = x + self.positional[:, :t, :]
        if move_ids is not None:
            x = x + self._embed_move_ids(move_ids)

        for attn, fc1, act, fc2 in self.layers:
            x = x + attn(self.norm(x))
            x = x + fc2(act(fc1(self.norm(x))))

        x = self.norm(x)
        return torch.where(mask.unsqueeze(-1), x, latents)


# ───────────────────── Policy / Value / Ponder head ──────────────────────────
class PolicyValuePonderHead(nn.Module):
    def __init__(self, latent_dim: int, predict_ponder: bool = False):
        super().__init__()
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.act = nn.ReLU()
        self.policy = nn.Linear(latent_dim, MOVE_VOCAB_SIZE)
        self.value = nn.Linear(latent_dim, 1)
        self.predict_ponder = predict_ponder
        if predict_ponder:
            self.ponder = nn.Linear(latent_dim, 1)

    def forward(
        self, latent: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = self.act(self.fc(latent))
        logits = self.policy(h)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        value = self.value(h).squeeze(-1)
        ponder_out = self.ponder(h).squeeze(-1) if self.predict_ponder else None
        return logits, value, ponder_out


# ───────────────────── Dataclass for pretraining batch ───────────────────────
@dataclass
class PretrainingBatch:
    latents: torch.Tensor  # (B, T, D)
    moves: torch.Tensor  # (B, T) int64
    mask: torch.Tensor  # (B, T) bool


# ───────────────────── ChessAgent wrapper ────────────────────────────────────
class ChessAgent:
    def __init__(self, latent_dim: int = 256, predict_ponder: bool = False, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.encoder = KingRelativeNNUE(latent_dim).to(self.device)
        self.world = JepaWorldModel(latent_dim).to(self.device)
        self.target_world = JepaWorldModel(latent_dim).to(self.device)
        self._init_target()
        self.predictor = PolicyValuePonderHead(latent_dim, predict_ponder).to(self.device)

    # ─── Momentum target initialisation / update ────────────────
    def _init_target(self):
        for p_t, p in zip(self.target_world.parameters(), self.world.parameters()):
            p_t.data.copy_(p.data)
            p_t.requires_grad_(False)

    def update_target(self, tau: float = 0.005):
        for p_t, p in zip(self.target_world.parameters(), self.world.parameters()):
            p_t.data.lerp_(p.data, tau)

    # ─── Pre‑training step (mask‑denoise, cosine loss) ───────────
    def pretrain_step(self, batch: PretrainingBatch) -> torch.Tensor:
        lat, mv, mask = batch.latents.to(self.device), batch.moves.to(self.device), batch.mask.to(self.device)
        masked_lat = lat.clone()
        masked_lat[mask] = 0.0

        pred = self.world(masked_lat, mv, mask)
        with torch.no_grad():
            tgt = self.target_world(lat, mv, mask)

        cos = F.cosine_similarity(pred, tgt, dim=-1)
        loss = (1.0 - cos)[mask].mean()
        return loss

    # ─── Fine‑tuning step (action‑conditioned) ──────────────────
    def fine_tune_step(self, latents: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        b, t, _ = latents.shape
        mask = torch.ones(b, t, dtype=torch.bool, device=latents.device)
        pred = self.world(latents, moves, mask)
        tgt = self.target_world(latents, moves, mask).detach()
        cos = F.cosine_similarity(pred, tgt, dim=-1)
        return (1.0 - cos).mean()

    # ─── Policy / value interface for search engines ─────────────
    def policy_value(
        self, board: chess.Board
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        latent, _ = self.encoder(board)
        legal = legal_moves_mask(board, latent.device)
        logits, value, ponder = self.predictor(latent, legal_mask=legal)
        return logits, value, ponder, legal

    # ─── RL update (simplified but complete) ─────────────────────
    def rl_update(self, trajectories: List[Dict[str, torch.Tensor]], optim: torch.optim.Optimizer, tau: float = 0.005):
        total_loss = torch.tensor(0.0, device=self.device)
        for traj in trajectories:
            lat_seq, mv_seq = traj["latents"].to(self.device), traj["moves"].to(self.device)
            pol_target, val_target = traj["policy"].to(self.device), traj["value"].to(self.device)
            ponder_tgt = traj.get("ponder")
            if ponder_tgt is not None:
                ponder_tgt = ponder_tgt.to(self.device)

            # policy/value/ponder losses
            logits, value, ponder = self.predictor(lat_seq, None)
            log_probs = F.log_softmax(logits, dim=-1)
            policy_loss = -(pol_target * log_probs).sum(-1).mean()
            value_loss = F.mse_loss(value, val_target)
            pv_loss = policy_loss + value_loss
            if ponder is not None and ponder_tgt is not None:
                pv_loss = pv_loss + F.mse_loss(ponder, ponder_tgt)
            total_loss = total_loss + pv_loss

            # model loss (predict next latent)
            if lat_seq.size(0) > 1:
                in_lat, tgt_lat = lat_seq[:-1].unsqueeze(0), lat_seq[1:].unsqueeze(0)
                in_mv = mv_seq[:-1].unsqueeze(0)
                mask = torch.ones_like(in_mv, dtype=torch.bool)
                pred = self.world(in_lat, in_mv, mask)
                cos = F.cosine_similarity(pred, tgt_lat, dim=-1)
                model_loss = (1 - cos).mean()
                total_loss = total_loss + model_loss

        total_loss = total_loss / max(len(trajectories), 1)
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        self.update_target(tau=tau)

    # ─── Convenience helpers ────────────────────────────────────
    def encode(self, board: chess.Board) -> torch.Tensor:
        return self.encoder(board)[0]

    def predict_next(self, latents: torch.Tensor, moves: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.world(latents, moves, mask)


# ───────────────────── Synthetic batch generator ─────────────────────────────
def random_pretraining_batch(
    batch: int = 4, seq: int = 16, latent_dim: int = 256, mask_prob: float = 0.15
) -> PretrainingBatch:
    lat = torch.randn(batch, seq, latent_dim)
    mv = torch.randint(0, MOVE_VOCAB_SIZE, (batch, seq))
    msk = torch.rand(batch, seq) < mask_prob
    return PretrainingBatch(lat, mv, msk)


# ───────────────────── Demo / smoke test ─────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = ChessAgent(device=device, predict_ponder=True)
    opt = torch.optim.Adam(agent.world.parameters(), lr=1e-3)

    print("=== Pre‑training JEPA for 5 steps ===")
    for step in range(5):
        batch = random_pretraining_batch()
        loss = agent.pretrain_step(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        agent.update_target()
        print(f"step {step}: loss = {loss.item():.4f}")

    board = chess.Board()
    logits, value, ponder, legal = agent.policy_value(board)
    probs = F.softmax(logits[legal], dim=-1)  # only legal moves
    best_idx = torch.argmax(probs).item()
    best_move_idx = torch.nonzero(legal)[best_idx].item()
    best_move = index_to_move(best_move_idx)

    print("\n=== Initial position ===")
    print("Best move (JEPA‑NNUE):", best_move.uci())
    print("Value estimate:", value.item())
    if ponder is not None:
        print("Suggested ponder (log‑ms):", ponder.item())


if __name__ == "__main__":
    main()
