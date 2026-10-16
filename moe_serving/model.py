"""Qwen-family expert-parallel models.

Qwen3 MoE, Qwen3.6, and Qwen3.8-Flash-Next share the same expert block:
a replicated router, a contiguous expert shard, and an optional replicated
shared expert. Qwen3.8 additionally has one Engram / PLE table. Attention
and the gated residual are separate backends; this module runs the expert
and Engram path that later models plug into.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from moe_serving.checkpoint import MappedCheckpoint
from moe_serving.distributed import Collective, SingleRank
from moe_serving.engram.dram import DramEngramTable
from moe_serving.engram.ple import NgramParameters, PleLayer
from moe_serving.moe import ExpertPlan, LocalSwiGLUExperts, MoELayer, RoutingStats, SharedSwiGLU

FAMILIES = ("qwen3_8_flash_next", "qwen3_moe", "qwen3_6")


@dataclass
class ModelSpec:
    family: str
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    num_layers: int
    vocab_size: int
    renormalize: bool
    rms_eps: float
    eos_token_id: int
    hc_count: int
    shared_intermediate: int | None
    ple_layer: int | None
    ngram_size: int
    heads_per_ngram: int
    prefix: str


class QwenExpertParallelModel(torch.nn.Module):
    def __init__(
        self,
        spec: ModelSpec,
        plan: ExpertPlan,
        layers: dict[int, MoELayer],
        ple: PleLayer | None = None,
        embed: torch.Tensor | None = None,
        attention: dict[int, torch.nn.Module] | None = None,
    ):
        super().__init__()
        self.spec = spec
        self.plan = plan
        self.moe = torch.nn.ModuleDict({str(layer_id): layer for layer_id, layer in layers.items()})
        self.ple = ple
        self.attention = torch.nn.ModuleDict(
            {str(layer_id): module for layer_id, module in (attention or {}).items()}
        )
        self.embed = torch.nn.Parameter(embed, requires_grad=False) if embed is not None else None

    @property
    def layer_ids(self) -> list[int]:
        return sorted(int(key) for key in self.moe.keys())

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, list[RoutingStats]]:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if self.embed is None:
            hidden = torch.zeros(
                *token_ids.shape,
                self.spec.hidden,
                device=token_ids.device,
                dtype=self._dtype(),
            )
        else:
            hidden = F.embedding(token_ids, self.embed)
        stats: list[RoutingStats] = []
        conv_state = None
        for layer_id in self.layer_ids:
            key = str(layer_id)
            if key in self.attention:
                hidden = self.attention[key](hidden)
            engram_seconds = 0.0
            if self.ple is not None and layer_id == self.spec.ple_layer:
                _sync(hidden)
                started = time.perf_counter()
                embeddings = self.ple.lookup(token_ids)
                hidden, conv_state = self.ple(hidden, embeddings, conv_state)
                _sync(hidden)
                engram_seconds = time.perf_counter() - started
            flat = hidden.reshape(-1, self.spec.hidden)
            _sync(flat)
            started = time.perf_counter()
            output, layer_stats = self.moe[str(layer_id)](flat)
            _sync(output)
            layer_stats.moe_seconds = time.perf_counter() - started
            layer_stats.engram_seconds = engram_seconds
            hidden = output.view_as(hidden)
            stats.append(layer_stats)
        return hidden, stats

    def _dtype(self) -> torch.dtype:
        layer = self.moe[str(self.layer_ids[0])]
        return layer.experts.down.dtype


def _sync(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def load_model(
    family: str,
    checkpoint: str,
    rank: int,
    ep_size: int,
    device: torch.device | str = "cpu",
    layers: list[int] | None = None,
    load_embeddings: bool = True,
    collective: Collective | None = None,
    log=None,
) -> QwenExpertParallelModel:
    if family not in FAMILIES:
        raise KeyError(f"unknown model {family}. Registered models: {', '.join(FAMILIES)}")
    log = log or (lambda *_args: None)
    raw = json.load(open(os.path.join(checkpoint, "config.json")))
    config = _text_config(raw)
    store = MappedCheckpoint(checkpoint)
    prefix = _detect_prefix(store.names())
    spec = _spec_from_config(family, config, store, prefix)
    selected = list(range(spec.num_layers) if layers is None else layers)
    plan = ExpertPlan(spec.num_experts, ep_size, rank)
    collective = collective or SingleRank()
    built: dict[int, MoELayer] = {}
    for layer_id in selected:
        log(f"rank {rank}: loading layer {layer_id} experts {plan.expert_start}:{plan.expert_end}")
        built[layer_id] = _load_moe_layer(store, spec, plan, layer_id, device, collective)
    ple = None
    if spec.ple_layer is not None and (layers is None or spec.ple_layer in selected):
        log(f"rank {rank}: mapping Engram table into DRAM")
        ple = _load_ple(store, spec, device)
    embed = None
    if load_embeddings:
        embed_name = _first(store, [f"{prefix}embed_tokens.weight", "embed_tokens.weight", "model.embed_tokens.weight"])
        if embed_name is not None:
            log(f"rank {rank}: loading token embeddings")
            embed = store.load(embed_name).to(device)
    return QwenExpertParallelModel(spec, plan, built, ple, embed).to(device)


def expert_shard_bytes(spec: ModelSpec, ep_size: int, num_layers: int, dtype_bytes: int = 2) -> int:
    local = spec.num_experts // ep_size
    per_expert = (2 * spec.intermediate * spec.hidden + spec.hidden * spec.intermediate) * dtype_bytes
    return local * per_expert * num_layers


def _text_config(raw: dict) -> dict:
    text = raw.get("text_config")
    if isinstance(text, dict):
        merged = dict(raw)
        merged.update(text)
        return merged
    return raw


def _detect_prefix(names: list[str]) -> str:
    for name in names:
        marker = ".layers."
        if marker in name:
            return name.split(marker)[0] + "."
        if name.startswith("layers."):
            return ""
    raise ValueError("checkpoint has no transformer layers")


def _spec_from_config(family: str, config: dict, store: MappedCheckpoint, prefix: str) -> ModelSpec:
    ple_layers = sorted(
        {
            int(match.group(1))
            for name in store.names()
            if (match := re.search(r"layers\.(\d+)\.ple\.", name))
        }
    )
    ple_layer = ple_layers[0] if ple_layers else None
    shared = config.get("shared_expert_intermediate_size")
    return ModelSpec(
        family=family,
        hidden=int(config["hidden_size"]),
        intermediate=int(config.get("moe_intermediate_size", config.get("intermediate_size"))),
        num_experts=int(config["num_experts"]),
        top_k=int(config.get("num_experts_per_tok", config.get("num_experts_per_token"))),
        num_layers=int(config["num_hidden_layers"]),
        vocab_size=int(config["vocab_size"]),
        renormalize=bool(config.get("norm_topk_prob", True)),
        rms_eps=float(config.get("rms_norm_eps", 1e-6)),
        eos_token_id=int(config.get("eos_token_id", 0)),
        hc_count=int(config.get("hc_count", 1)),
        shared_intermediate=int(shared) if shared else None,
        ple_layer=ple_layer,
        ngram_size=int(config.get("ngram_size", 3)),
        heads_per_ngram=int(config.get("heads_per_ngram", 8)),
        prefix=prefix,
    )


def _load_moe_layer(store, spec: ModelSpec, plan: ExpertPlan, layer_id: int, device, collective) -> MoELayer:
    prefix = f"{spec.prefix}layers.{layer_id}.mlp."
    gate_name = _first(store, [prefix + "gate.weight"])
    gate_up_name = _first(store, [prefix + "experts.gate_up_proj", prefix + "experts.gate_up_proj.weight"])
    down_name = _first(store, [prefix + "experts.down_proj", prefix + "experts.down_proj.weight"])
    if gate_up_name and down_name:
        gate_up = store.slice_experts(gate_up_name, plan.expert_start, plan.expert_end)
        down = store.slice_experts(down_name, plan.expert_start, plan.expert_end)
    else:
        gate_up, down = _stack_per_expert(store, prefix, plan)
    experts = LocalSwiGLUExperts(gate_up.to(device), down.to(device))
    shared = _load_shared(store, prefix, device)
    return MoELayer(
        gate_weight=store.load(gate_name).to(device),
        experts=experts,
        plan=plan,
        top_k=spec.top_k,
        renormalize=spec.renormalize,
        shared=shared,
        collective=collective,
        layer_id=layer_id,
    )


def _stack_per_expert(store: MappedCheckpoint, prefix: str, plan: ExpertPlan):
    gates, ups, downs = [], [], []
    for expert_id in range(plan.expert_start, plan.expert_end):
        stem = f"{prefix}experts.{expert_id}."
        gates.append(store.load(_require(store, [stem + "gate_proj.weight", stem + "gate_proj"])))
        ups.append(store.load(_require(store, [stem + "up_proj.weight", stem + "up_proj"])))
        downs.append(store.load(_require(store, [stem + "down_proj.weight", stem + "down_proj"])))
    gate_up = torch.cat([torch.stack(gates), torch.stack(ups)], dim=1)
    return gate_up, torch.stack(downs)


def _load_shared(store: MappedCheckpoint, prefix: str, device) -> SharedSwiGLU | None:
    stem = prefix + "shared_expert."
    gate = _first(store, [stem + "gate_proj.weight"])
    if gate is None:
        return None
    return SharedSwiGLU(
        store.load(gate).to(device),
        store.load(stem + "up_proj.weight").to(device),
        store.load(stem + "down_proj.weight").to(device),
        store.load(prefix + "shared_expert_gate.weight").to(device),
    )


def _load_ple(store: MappedCheckpoint, spec: ModelSpec, device) -> PleLayer:
    prefix = f"{spec.prefix}layers.{spec.ple_layer}.ple."
    shard_names = sorted(
        (name for name in store.names() if name.startswith(prefix + "ple_embedding.ngram_embedding.shard_")),
        key=lambda name: int(name.split("shard_")[1].split(".")[0]),
    )
    table = DramEngramTable(store, shard_names)
    embedding = prefix + "ple_embedding."
    parameters = NgramParameters(
        multipliers=store.load(embedding + "layer_multipliers").to(torch.int64).cpu(),
        vocab_sizes=store.load(embedding + "ngram_heads_vocab_sizes").to(torch.int64).cpu(),
        offsets=store.load(embedding + "ngram_heads_offsets").to(torch.int64).cpu(),
        heads_per_ngram=spec.heads_per_ngram,
        eos_token_id=spec.eos_token_id,
    )
    return PleLayer(
        table=table,
        parameters=parameters,
        key_proj=store.load(prefix + "key_proj.weight").to(device),
        value_proj=store.load(prefix + "value_proj.weight").to(device),
        norm_query=store.load(prefix + "norm_query.weight").to(device),
        norm_key=store.load(prefix + "norm_key.weight").to(device),
        norm_conv=store.load(prefix + "norm_conv.weight").to(device),
        conv_weight=store.load(prefix + "conv1d.weight").to(device),
        hc_count=spec.hc_count,
        eps=spec.rms_eps,
    )


def _first(store: MappedCheckpoint, names: list[str]) -> str | None:
    for name in names:
        if name in store.weight_map:
            return name
    return None


def _require(store: MappedCheckpoint, names: list[str]) -> str:
    found = _first(store, names)
    if found is None:
        raise KeyError(names[0])
    return found
