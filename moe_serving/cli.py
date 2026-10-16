"""Profile expert-parallel routing.

Example, four GPUs, Engram left in DRAM, first two layers::

    torchrun --nproc_per_node=4 -m moe_serving.cli profile \\
        --model qwen3_8_flash_next \\
        --checkpoint /home/sky/MoE_Serving_Profile/LLMs/Qwen3.8-Flash-Next \\
        --layers 0,1 --seq-len 128
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from moe_serving.distributed import collective_for, init_process_group
from moe_serving.engine import Engine
from moe_serving.model import FAMILIES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Expert-parallel MoE inference and routing profile")
    parser.add_argument("--model", required=True, choices=FAMILIES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layers", default=None, help="comma-separated layer ids; default is every layer")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--token-ids", default=None, help="JSON file of integer token ids, shape [batch, time] or [time]")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--device", default=None, help="cuda or cpu; default is cuda when it is available")
    args = parser.parse_args(argv)

    if torch.cuda.is_available() and args.device != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    rank, world = init_process_group()
    layers = None if args.layers is None else [int(part) for part in args.layers.split(",") if part]
    device = args.device
    if device is None:
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}" if torch.cuda.is_available() else "cpu"
    engine = Engine.from_pretrained(
        args.model,
        args.checkpoint,
        rank=rank,
        ep_size=world,
        device=device,
        layers=layers,
        load_embeddings=not args.no_embed,
        collective=collective_for(world),
        log=print if rank == 0 else None,
    )
    if rank == 0:
        print(engine.describe())
    token_ids = _token_ids(args, engine, rank, world)
    rows = engine.profile(token_ids)
    if rank == 0:
        for row in rows:
            print(json.dumps(row))


def _token_ids(args, engine: Engine, rank: int, world: int) -> torch.Tensor:
    if args.token_ids:
        ids = torch.tensor(json.load(open(args.token_ids)), dtype=torch.int64)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        return ids
    vocab = engine.model.spec.vocab_size
    if rank == 0:
        ids = torch.randint(0, vocab, (args.batch, args.seq_len), dtype=torch.int64)
    else:
        ids = torch.empty(args.batch, args.seq_len, dtype=torch.int64)
    if world > 1 and torch.distributed.is_initialized():
        torch.distributed.broadcast(ids, src=0)
    return ids


if __name__ == "__main__":
    main()
