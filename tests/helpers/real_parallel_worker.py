"""Compare every real checkpoint layer with native Transformers using 4 GPUs.

Weights are streamed by layer to bound GPU memory. All layers participate in
end-to-end teacher-forced prefill/decode, with batch and solo cases. The native
reference has full attention heads and all experts; only the tested model uses
TP/EP. No model/source/checkpoint files are changed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from moe_serving import gpt_oss_parallel, qwen3_parallel


def source_name(name, layer):
    return name.replace("model.layers.0.", f"model.layers.{layer}.")


@contextmanager
def layer_checkpoint(checkpoint, raw_config, weight_map, layer, backend):
    """A read-only alias view, renaming just the selected layer to layer zero."""
    config = copy.deepcopy(raw_config)
    config["num_hidden_layers"] = 1
    if "layer_types" in config:
        config["layer_types"] = [raw_config["layer_types"][layer]]
    if "mlp_only_layers" in config:
        config["mlp_only_layers"] = (
            [0] if layer in raw_config["mlp_only_layers"] else []
        )
    # These production Qwen3 checkpoints have sparse experts at every layer.
    if config.get("decoder_sparse_step", 1) != 1:
        raise ValueError("Streaming validation requires decoder_sparse_step=1")
    aliased = {}
    prefix = f"model.layers.{layer}."
    for name, file in weight_map.items():
        if name.startswith(prefix):
            name = name.replace(prefix, "model.layers.0.", 1)
        elif name.startswith("model.layers."):
            continue
        aliased[name] = str((checkpoint / file).resolve())
    original_open = backend.safe_open

    class Reader:
        def __init__(self, *args, **kwargs):
            self.context = original_open(*args, **kwargs)

        def __enter__(self):
            self.handle = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def get_slice(self, name):
            return self.handle.get_slice(source_name(name, layer))

        def get_tensor(self, name):
            return self.handle.get_tensor(source_name(name, layer))

    with tempfile.TemporaryDirectory(prefix="moe-validation-") as directory:
        root = Path(directory)
        (root / "config.json").write_text(json.dumps(config))
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": aliased})
        )
        backend.safe_open = Reader
        try:
            yield root, config
        finally:
            backend.safe_open = original_open


class NativeReader:
    """Independent native-model load, with separate Qwen FP8 dequantization."""

    def __init__(self, checkpoint, weight_map, config, device, dtype):
        self.checkpoint, self.weight_map, self.config = checkpoint, weight_map, config
        self.device, self.dtype = device, dtype
        self.handles = {}
        self.roots = {}

    def close(self):
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()

    def tensor(self, name):
        file = self.weight_map[name]
        if file not in self.handles:
            self.handles[file] = safe_open(self.checkpoint / file, framework="pt")
        return self.handles[file].get_tensor(name)

    def linear(self, name):
        value = self.tensor(name).to(self.device)
        scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
        if scale_name in self.weight_map:
            # Reshape/broadcast implementation independent of the backend's
            # repeat_interleave dequantization and attention shard slicing.
            scales = self.tensor(scale_name).to(self.device).float()
            br, bc = self.config["quantization_config"]["weight_block_size"]
            rows, cols = value.shape
            nr, nc = scales.shape
            value = F.pad(value.float(), (0, nc * bc - cols, 0, nr * br - rows))
            value = value.reshape(nr, br, nc, bc) * scales[:, None, :, None]
            value = value.reshape(nr * br, nc * bc)[:rows, :cols]
        return value.to(self.dtype)

    def model(self, layer, small_config, family):
        from transformers import (
            GptOssConfig,
            GptOssForCausalLM,
            Qwen3MoeConfig,
            Qwen3MoeForCausalLM,
        )
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeRotaryEmbedding,
        )

        cls, config_cls, rotary = (
            (GptOssForCausalLM, GptOssConfig, GptOssRotaryEmbedding)
            if family == "gpt_oss"
            else (Qwen3MoeForCausalLM, Qwen3MoeConfig, Qwen3MoeRotaryEmbedding)
        )
        config = config_cls.from_dict(small_config)
        config._attn_implementation = config._experts_implementation = "eager"
        with torch.device("meta"):
            model = cls(config)
        with torch.device(self.device):
            model.model.rotary_emb = rotary(config)
        for name, parameter in list(model.named_parameters()):
            actual = source_name(name, layer)
            if not name.startswith("model.layers.") and name in self.roots:
                value = self.roots[name]
            elif family == "qwen3_moe" and ".mlp.experts." in name:
                value = torch.empty(
                    parameter.shape, device=self.device, dtype=self.dtype
                )
                prefix, proj = actual.rsplit(".", 1)
                if actual in self.weight_map:
                    value.copy_(self.tensor(actual).to(self.device, self.dtype))
                else:
                    for expert in range(config.num_experts):
                        if proj == "gate_up_proj":
                            for slot, projection in enumerate(("gate_proj", "up_proj")):
                                width = config.moe_intermediate_size
                                value[expert, slot * width : (slot + 1) * width].copy_(
                                    self.linear(
                                        f"{prefix}.{expert}.{projection}.weight"
                                    )
                                )
                        else:
                            value[expert].copy_(
                                self.linear(f"{prefix}.{expert}.down_proj.weight")
                            )
            elif (
                family == "gpt_oss"
                and ".mlp.experts." in name
                and name.endswith(("gate_up_proj", "down_proj"))
            ):
                if actual in self.weight_map:
                    value = self.tensor(actual).to(self.device, self.dtype)
                else:
                    from transformers.integrations.mxfp4 import (
                        _convert_moe_packed_tensors,
                    )

                    blocks = self.tensor(actual + "_blocks")
                    scales = self.tensor(actual + "_scales")
                    value = torch.empty(
                        parameter.shape, device=self.device, dtype=self.dtype
                    )
                    for expert in range(config.num_local_experts):
                        value[expert].copy_(
                            _convert_moe_packed_tensors(
                                blocks[expert : expert + 1].to(self.device),
                                scales[expert : expert + 1].to(self.device),
                                dtype=self.dtype,
                            )[0]
                        )
            else:
                value = self.linear(actual)
            parent, attr = name.rsplit(".", 1)
            model.get_submodule(parent)._parameters[attr] = nn.Parameter(
                value, requires_grad=False
            )
            if not name.startswith("model.layers."):
                self.roots[name] = value
        assert not any(
            t.is_meta for t in list(model.parameters()) + list(model.buffers())
        )
        return model.eval()


def compare(actual, expected, atol, rtol):
    a, b = actual.detach().float(), expected.detach().float()
    delta = (a - b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {
        "pass": finite and bool((delta <= atol + rtol * b.abs()).all()),
        "max_abs": float(delta.max()),
        "bitwise_exact": bool(torch.equal(actual, expected)),
        "relative_l2": float(
            torch.linalg.vector_norm(a - b)
            / torch.linalg.vector_norm(b).clamp_min(1e-12)
        ),
    }


def prepare_cases(rows, steps, vocab_size, device):
    cases = []
    for name, inputs in [("batch", rows), ("solo", rows[:1])]:
        width = max(map(len, inputs))
        ids = torch.tensor(
            [[0] * (width - len(row)) + row for row in inputs], device=device
        )
        mask = torch.tensor(
            [[0] * (width - len(row)) + [1] * len(row) for row in inputs], device=device
        )
        case = {"name": name, "ids": [ids], "mask": [mask]}
        for step in range(steps):
            # Fixed teacher forcing makes all implementations see identical data.
            token = torch.full(
                (len(inputs), 1), (17 + step) % vocab_size, device=device
            )
            case["ids"].append(token)
            mask = torch.cat(
                (mask, torch.ones(len(inputs), 1, device=device, dtype=mask.dtype)), -1
            )
            case["mask"].append(mask)
        case["positions"] = [
            (m.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1] :]
            for ids, m in zip(case["ids"], case["mask"])
        ]
        cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--dump-layer", type=int)
    parser.add_argument(
        "--parallel-numerics", choices=["reference", "fast"], default="reference"
    )
    parser.add_argument(
        "--token-ids", help="Two JSON token rows; otherwise tokenize two short prompts"
    )
    parser.add_argument(
        "--max-layers", type=int, help="Debug only; omitted for full-model validation"
    )
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise ValueError("Run with torchrun --nproc-per-node=4")
    torch.cuda.set_device(rank)
    device, dtype = torch.device("cuda", rank), getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    checkpoint = args.checkpoint.resolve()
    raw = json.loads((checkpoint / "config.json").read_text())
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    family = raw["model_type"]
    backend = gpt_oss_parallel if family == "gpt_oss" else qwen3_parallel
    backend.ensure_transformers_5200()
    # A production load creates groups once. Repeated layer loads must reuse
    # them: NCCL communicators otherwise accumulate until device memory fills.
    shared_parallel = backend.ParallelGroup(world)

    def parallel_group(size):
        if size != world:
            raise ValueError("This validator requires TP=EP=WORLD_SIZE=4")
        return shared_parallel

    backend.ParallelGroup = parallel_group
    if args.token_ids:
        rows = json.loads(args.token_ids)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        rows = [
            tokenizer.encode(text, add_special_tokens=False)[:16]
            for text in [
                "What is two plus three? Give a short answer.",
                "Explain MoE briefly.",
            ]
        ]
    cases = prepare_cases(rows, args.decode_steps, raw["vocab_size"], device)
    atol, rtol = (1e-3, 2e-4) if dtype == torch.float32 else (0.25, 0.03)
    reader = (
        NativeReader(checkpoint, weight_map, raw, device, dtype) if rank == 0 else None
    )
    report = {
        "checkpoint": str(checkpoint),
        "model_type": family,
        "dtype": args.dtype,
        "tp_size": 4,
        "parallel_numerics": args.parallel_numerics,
        "ep_size": 4,
        "atol": atol,
        "rtol": rtol,
        "token_rows": rows,
        "decode_steps": args.decode_steps,
        "method": "all-layer streamed native reference, teacher forced",
        "layers": [],
        "logits": [],
        "full_model": args.max_layers is None,
    }
    baseline = {
        "checkpoint": str(checkpoint),
        "dtype": args.dtype,
        "parallel_numerics": args.parallel_numerics,
        "token_rows": rows,
        "decode_steps": args.decode_steps,
        "routing": {},
        "logits": {},
    }
    count = min(raw["num_hidden_layers"], args.max_layers or raw["num_hidden_layers"])
    start = time.perf_counter()
    with torch.inference_mode():
        for layer in range(count):
            with layer_checkpoint(checkpoint, raw, weight_map, layer, backend) as (
                view,
                small_config,
            ):
                actual = backend.load_parallel_model(
                    view,
                    4,
                    4,
                    device,
                    dtype,
                    "mxfp4" if family == "gpt_oss" else "fp8",
                    numerics=args.parallel_numerics,
                )
            expected = reader.model(layer, small_config, family) if rank == 0 else None
            actual_norm, actual_head = actual.model.norm, actual.lm_head
            actual.model.norm, actual.lm_head = nn.Identity(), nn.Identity()
            if rank == 0:
                expected_norm, expected_head = expected.model.norm, expected.lm_head
                expected.model.norm, expected.lm_head = nn.Identity(), nn.Identity()
            trace = backend.RoutingRecorder(actual)
            ref_trace = backend.RoutingRecorder(expected) if rank == 0 else None
            results = []
            if args.dump_layer == layer and rank == 0:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        case["name"]: {
                            "actual": [h.cpu() for h in case["hidden"]],
                            "reference": [h.cpu() for h in case["reference"]],
                        }
                        for case in cases
                    },
                    args.report.with_suffix(".inputs.pt"),
                )
            for case in cases:
                if layer == 0:
                    case["hidden"] = [
                        actual.model.embed_tokens(ids) for ids in case["ids"]
                    ]
                    if rank == 0:
                        case["reference"] = [
                            expected.model.embed_tokens(ids) for ids in case["ids"]
                        ]
                cache = ref_cache = None
                for step, (ids, mask, positions) in enumerate(
                    zip(case["ids"], case["mask"], case["positions"])
                ):
                    trace.begin(ids, mask, positions)
                    out = actual(
                        inputs_embeds=case["hidden"][step],
                        attention_mask=mask,
                        position_ids=positions,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = out.past_key_values
                    case["hidden"][step] = out.logits
                    if rank == 0:
                        ref_trace.begin(ids, mask, positions)
                        ref = expected(
                            inputs_embeds=case["reference"][step],
                            attention_mask=mask,
                            position_ids=positions,
                            past_key_values=ref_cache,
                            use_cache=True,
                        )
                        ref_cache = ref.past_key_values
                        case["reference"][step] = ref.logits
                        valid = mask[:, -ids.shape[1] :].bool()
                        metrics = compare(
                            out.logits[valid], ref.logits[valid], atol, rtol
                        )
                        a, b = trace.records[-1], ref_trace.records[-1]
                        baseline["routing"][f"{layer}:{case['name']}:{step}"] = {
                            "indices": b["expert_indices"][valid.cpu()],
                            "scores": b["routing_weights"][valid.cpu()],
                        }
                        ai, order_a = a["expert_indices"][valid.cpu()].sort(-1)
                        bi, order_b = b["expert_indices"][valid.cpu()].sort(-1)
                        scores = compare(
                            a["routing_weights"][valid.cpu()].gather(-1, order_a),
                            b["routing_weights"][valid.cpu()].gather(-1, order_b),
                            atol,
                            rtol,
                        )
                        results.append(
                            {
                                "case": case["name"],
                                "step": step,
                                "hidden": metrics,
                                "routing_exact": bool(
                                    torch.equal(
                                        a["expert_indices"][valid.cpu()],
                                        b["expert_indices"][valid.cpu()],
                                    )
                                ),
                                "routing_set_exact": bool(torch.equal(ai, bi)),
                                "routing_matching_slots": float(
                                    (ai == bi).float().mean()
                                ),
                                "scores": scores,
                            }
                        )
                    assert (
                        cache.layers[0].keys.shape[1] == raw["num_key_value_heads"] // 4
                    )
                    cache_metrics = []
                    for attr in ("keys", "values"):
                        local = getattr(cache.layers[0], attr)
                        full_shape = list(local.shape)
                        full_shape[1] *= 4
                        full = (
                            getattr(ref_cache.layers[0], attr).contiguous()
                            if rank == 0
                            else torch.empty(full_shape, device=device, dtype=dtype)
                        )
                        dist.broadcast(full, src=0)
                        reference_slice = full[
                            :, rank * local.shape[1] : (rank + 1) * local.shape[1]
                        ]
                        key_mask = mask[:, -local.shape[2] :].bool()
                        m = compare(
                            local.transpose(1, 2)[key_mask],
                            reference_slice.transpose(1, 2)[key_mask],
                            atol,
                            rtol,
                        )
                        summary = torch.tensor(
                            [
                                m["max_abs"],
                                m["relative_l2"],
                                float(not m["pass"]),
                                float(not m["bitwise_exact"]),
                            ],
                            device=device,
                        )
                        dist.all_reduce(summary, op=dist.ReduceOp.MAX)
                        cache_metrics.append(
                            {
                                "kind": attr,
                                "max_abs": float(summary[0]),
                                "relative_l2": float(summary[1]),
                                "pass": float(summary[2]) == 0,
                                "bitwise_exact": float(summary[3]) == 0,
                            }
                        )
                        del full, reference_slice, local
                    if rank == 0:
                        results[-1]["cache"] = cache_metrics
                del out, cache
                if rank == 0:
                    del ref, ref_cache
            trace.close()
            if ref_trace is not None:
                ref_trace.close()
            if rank == 0:
                report["layers"].append({"layer": layer, "results": results})
                report["status"] = "running"
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "layer": layer,
                            "total": count,
                            "max_abs": max(r["hidden"]["max_abs"] for r in results),
                            "numerical_pass": all(
                                r["hidden"]["pass"]
                                and r["scores"]["pass"]
                                and all(m["pass"] for m in r["cache"])
                                for r in results
                            ),
                            "routing_exact": all(r["routing_exact"] for r in results),
                            "seconds": round(time.perf_counter() - start, 1),
                        }
                    ),
                    flush=True,
                )
            if layer == count - 1:
                for case in cases:
                    case["logits"] = [
                        actual_head(actual_norm(h[:, -1:])) for h in case["hidden"]
                    ]
                    if rank == 0:
                        case["ref_logits"] = [
                            expected_head(expected_norm(h[:, -1:]))
                            for h in case["reference"]
                        ]
                        for step, (a, b) in enumerate(
                            zip(case["logits"], case["ref_logits"])
                        ):
                            baseline["logits"][f"{case['name']}:{step}"] = (
                                b.detach().float().cpu()
                            )
                            report["logits"].append(
                                {
                                    "case": case["name"],
                                    "step": step,
                                    **compare(a, b, atol, rtol),
                                    "top1_exact": bool(
                                        torch.equal(a.argmax(-1), b.argmax(-1))
                                    ),
                                    "actual_top1": a.argmax(-1).flatten().tolist(),
                                    "reference_top1": b.argmax(-1).flatten().tolist(),
                                }
                            )
                if rank == 0:
                    report["batch_vs_solo"] = [
                        compare(a[:1], b, atol, rtol)
                        for a, b in zip(cases[0]["logits"], cases[1]["logits"])
                    ]
            del actual, expected, actual_norm, actual_head, trace, ref_trace
            if rank == 0:
                del expected_norm, expected_head
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
    if rank == 0:
        report["seconds"] = time.perf_counter() - start
        report["numerical_pass"] = all(
            r["hidden"]["pass"]
            and r["scores"]["pass"]
            and all(m["pass"] for m in r["cache"])
            for layer in report["layers"]
            for r in layer["results"]
        ) and all(r["pass"] for r in report["logits"])
        report["routing_exact"] = all(
            r["routing_exact"] for layer in report["layers"] for r in layer["results"]
        )
        report["top1_exact"] = all(r["top1_exact"] for r in report["logits"])
        report["bitwise_exact"] = all(
            r["hidden"]["bitwise_exact"]
            and r["scores"]["bitwise_exact"]
            and all(m["bitwise_exact"] for m in r["cache"])
            for layer in report["layers"]
            for r in layer["results"]
        ) and all(r["bitwise_exact"] for r in report["logits"])
        report["batch_vs_solo_pass"] = all(r["pass"] for r in report["batch_vs_solo"])
        report["batch_semantics_exact"] = all(
            torch.equal(a[:1].float() - b.float(), ra[:1].float() - rb.float())
            for a, b, ra, rb in zip(
                cases[0]["logits"],
                cases[1]["logits"],
                cases[0]["ref_logits"],
                cases[1]["ref_logits"],
            )
        )
        report["native_reference_pass"] = (
            report["numerical_pass"]
            and report["routing_exact"]
            and report["top1_exact"]
            and report["batch_semantics_exact"]
        )
        if args.parallel_numerics == "reference":
            report["native_reference_pass"] = (
                report["native_reference_pass"] and report["bitwise_exact"]
            )
        torch.save(baseline, args.report.with_suffix(".reference.pt"))
        report["reference_file"] = str(args.report.with_suffix(".reference.pt"))
        report["status"] = "complete"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in {"layers", "logits", "token_rows", "batch_vs_solo"}
                }
            ),
            flush=True,
        )
        reader.close()
    failed = torch.zeros((), device=device, dtype=torch.int32)
    if rank == 0:
        passed = (
            report["numerical_pass"]
            and report["routing_exact"]
            and report["top1_exact"]
        )
        passed = passed and report.get(
            "batch_semantics_exact", report.get("batch_vs_solo_pass", True)
        )
        if args.parallel_numerics == "reference":
            passed = passed and report["bitwise_exact"]
        failed.fill_(int(not passed))
    dist.broadcast(failed, src=0)
    exit_code = int(failed.item())
    dist.destroy_process_group()
    if exit_code:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
