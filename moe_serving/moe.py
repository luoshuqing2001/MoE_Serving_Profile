"""Expert-parallel token routing for a SwiGLU MoE block.

Each rank owns a contiguous slice of experts. A token is sent once per
selected expert, the owner runs that expert, and the weighted outputs are
added back on the token's rank. The shared expert, when the model has one,
stays replicated and runs on the local tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from moe_serving.distributed import Collective, SingleRank


@dataclass
class ExpertPlan:
    num_experts: int
    ep_size: int
    rank: int

    def __post_init__(self) -> None:
        if self.num_experts % self.ep_size:
            raise ValueError(
                f"{self.num_experts} experts do not divide across {self.ep_size} ranks"
            )
        self.experts_per_rank = self.num_experts // self.ep_size
        self.expert_start = self.rank * self.experts_per_rank
        self.expert_end = self.expert_start + self.experts_per_rank

    def owner(self, expert_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dest = torch.div(expert_ids, self.experts_per_rank, rounding_mode="floor")
        local = expert_ids - dest * self.experts_per_rank
        return dest, local


@dataclass
class RoutingStats:
    counts: torch.Tensor
    layer: int = 0
    moe_seconds: float = 0.0
    engram_seconds: float = 0.0

    @property
    def expert_load_cv(self) -> float:
        mean = self.counts.float().mean()
        if float(mean) == 0:
            return 0.0
        return float(self.counts.float().std(unbiased=False) / mean)

    @property
    def expert_dead_fraction(self) -> float:
        return float((self.counts == 0).float().mean())


class LocalSwiGLUExperts(torch.nn.Module):
    """Fused gate/up expert weights owned by one rank.

    ``gate_up`` is ``[local_experts, 2 * intermediate, hidden]`` with the
    gate half first. ``down`` is ``[local_experts, hidden, intermediate]``.
    """

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor):
        super().__init__()
        self.gate_up = torch.nn.Parameter(gate_up, requires_grad=False)
        self.down = torch.nn.Parameter(down, requires_grad=False)
        self.hidden = down.shape[1]
        self.intermediate = down.shape[2]

    def forward(self, hidden: torch.Tensor, local_expert_ids: torch.Tensor) -> torch.Tensor:
        if hidden.shape[0] == 0:
            return hidden.new_empty(0, self.hidden)
        out = hidden.new_empty(hidden.shape[0], self.hidden)
        for expert in local_expert_ids.unique():
            mask = local_expert_ids == expert
            gate, up = F.linear(hidden[mask], self.gate_up[int(expert)]).chunk(2, dim=-1)
            out[mask] = F.linear(F.silu(gate) * up, self.down[int(expert)])
        return out


class SharedSwiGLU(torch.nn.Module):
    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, score: torch.Tensor):
        super().__init__()
        self.gate = torch.nn.Parameter(gate, requires_grad=False)
        self.up = torch.nn.Parameter(up, requires_grad=False)
        self.down = torch.nn.Parameter(down, requires_grad=False)
        self.score = torch.nn.Parameter(score, requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        activated = F.silu(F.linear(hidden, self.gate)) * F.linear(hidden, self.up)
        return torch.sigmoid(F.linear(hidden, self.score)) * F.linear(activated, self.down)


def route(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.softmax(F.linear(hidden.float(), gate_weight.float()), dim=-1)
    weight, index = torch.topk(scores, top_k, dim=-1)
    if renormalize:
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return index.to(torch.int64), weight.to(hidden.dtype)


def routed_forward(
    hidden: torch.Tensor,
    expert_index: torch.Tensor,
    expert_weight: torch.Tensor,
    experts: LocalSwiGLUExperts,
    plan: ExpertPlan,
    collective: Collective,
) -> torch.Tensor:
    """Dispatch local tokens, run the owned experts, and combine."""
    if plan.ep_size > 1 and isinstance(collective, SingleRank):
        raise RuntimeError("expert parallel needs a process group; launch with torchrun")
    flat_index = expert_index.reshape(-1)
    flat_weight = expert_weight.reshape(-1)
    token_index = torch.arange(hidden.shape[0], device=hidden.device).repeat_interleave(expert_index.shape[1])
    dest, local_expert = plan.owner(flat_index)
    order = torch.argsort(dest, stable=True)
    send_counts = torch.bincount(dest, minlength=plan.ep_size)
    send_hidden = hidden[token_index[order]]
    send_expert = local_expert[order].to(torch.int64)

    recv_hidden, recv_counts = collective.exchange(send_hidden, send_counts)
    recv_expert, _recv_counts = collective.exchange(send_expert.unsqueeze(-1), send_counts)
    computed = experts(recv_hidden, recv_expert.squeeze(-1))
    returned, _back_counts = collective.exchange(computed, recv_counts)

    combined = hidden.new_zeros(hidden.shape)
    combined.index_add_(0, token_index[order], returned * flat_weight[order].unsqueeze(-1).to(returned.dtype))
    return combined


class MoELayer(torch.nn.Module):
    def __init__(
        self,
        gate_weight: torch.Tensor,
        experts: LocalSwiGLUExperts,
        plan: ExpertPlan,
        top_k: int,
        renormalize: bool = True,
        shared: SharedSwiGLU | None = None,
        collective: Collective | None = None,
        layer_id: int = 0,
    ):
        super().__init__()
        self.gate_weight = torch.nn.Parameter(gate_weight, requires_grad=False)
        self.experts = experts
        self.shared = shared
        self.plan = plan
        self.top_k = top_k
        self.renormalize = renormalize
        self.collective = collective or SingleRank()
        self.layer_id = layer_id

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, RoutingStats]:
        index, weight = route(hidden, self.gate_weight, self.top_k, self.renormalize)
        output = routed_forward(hidden, index, weight, self.experts, self.plan, self.collective)
        if self.shared is not None:
            output = output + self.shared(hidden)
        counts = torch.bincount(index.reshape(-1).cpu(), minlength=self.plan.num_experts)
        return output, RoutingStats(counts=counts, layer=self.layer_id)


def simulate_ranks(
    hiddens: list[torch.Tensor],
    gates: torch.Tensor,
    expert_banks: list[LocalSwiGLUExperts],
    top_k: int,
    renormalize: bool = True,
    shared: list[SharedSwiGLU] | None = None,
) -> list[torch.Tensor]:
    """Run every rank inside one process. Used to check the all-to-all math."""
    from moe_serving.distributed import exchange_inplace

    ep_size = len(hiddens)
    plans = [ExpertPlan(gates.shape[0], ep_size, rank) for rank in range(ep_size)]
    packed = []
    for hidden, plan in zip(hiddens, plans):
        index, weight = route(hidden, gates, top_k, renormalize)
        flat_index = index.reshape(-1)
        flat_weight = weight.reshape(-1)
        token_index = torch.arange(hidden.shape[0]).repeat_interleave(index.shape[1])
        dest, local_expert = plan.owner(flat_index)
        order = torch.argsort(dest, stable=True)
        send_counts = torch.bincount(dest, minlength=ep_size)
        packed.append(
            {
                "hidden": hidden,
                "order": order,
                "token_index": token_index,
                "weight": flat_weight,
                "send_hidden": hidden[token_index[order]],
                "send_expert": local_expert[order].to(torch.int64).unsqueeze(-1),
                "send_counts": send_counts,
            }
        )
    recv_h = exchange_inplace([item["send_hidden"] for item in packed], [item["send_counts"] for item in packed])
    recv_e = exchange_inplace([item["send_expert"] for item in packed], [item["send_counts"] for item in packed])
    computed = []
    counts = []
    for rank, ((rows, _counts), (experts_ids, recv_counts)) in enumerate(zip(recv_h, recv_e)):
        computed.append(expert_banks[rank](rows, experts_ids.squeeze(-1)))
        counts.append(recv_counts)
    returned = exchange_inplace(computed, counts)
    outputs = []
    for item, (rows, _counts) in zip(packed, returned):
        combined = item["hidden"].new_zeros(item["hidden"].shape)
        combined.index_add_(
            0,
            item["token_index"][item["order"]],
            rows * item["weight"][item["order"]].unsqueeze(-1).to(rows.dtype),
        )
        rank = len(outputs)
        if shared is not None:
            combined = combined + shared[rank](item["hidden"])
        outputs.append(combined)
    return outputs
