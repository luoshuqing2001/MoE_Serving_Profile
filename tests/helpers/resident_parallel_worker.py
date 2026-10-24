"""Compare the production, fully resident 4-rank loader with a saved native baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from real_parallel_worker import compare, prepare_cases

from moe_serving import gpt_oss_parallel, qwen3_parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    baseline = torch.load(args.reference, map_location="cpu", weights_only=True)
    checkpoint = Path(baseline["checkpoint"])
    raw = json.loads((checkpoint / "config.json").read_text())
    backend = gpt_oss_parallel if raw["model_type"] == "gpt_oss" else qwen3_parallel
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = getattr(torch, baseline["dtype"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    model = backend.load_parallel_model(
        checkpoint,
        4,
        4,
        device,
        dtype,
        "mxfp4" if raw["model_type"] == "gpt_oss" else "fp8",
        numerics=baseline.get("parallel_numerics", "reference"),
    )
    cases = prepare_cases(
        baseline["token_rows"], baseline["decode_steps"], raw["vocab_size"], device
    )
    atol, rtol = (1e-3, 2e-4) if dtype == torch.float32 else (0.25, 0.03)
    report = {
        "checkpoint": str(checkpoint),
        "dtype": baseline["dtype"],
        "tp_size": 4,
        "parallel_numerics": baseline.get("parallel_numerics", "reference"),
        "ep_size": 4,
        "method": "production fully resident model versus streamed native baseline",
        "logits": [],
        "routing": [],
    }
    with torch.inference_mode():
        for case in cases:
            trace = backend.RoutingRecorder(model)
            cache = None
            for step, (ids, mask, pos) in enumerate(
                zip(case["ids"], case["mask"], case["positions"])
            ):
                trace.begin(ids, mask, pos)
                out = model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=pos,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = out.past_key_values
                if rank == 0:
                    expected = baseline["logits"][f"{case['name']}:{step}"].to(device)
                    report["logits"].append(
                        {
                            "case": case["name"],
                            "step": step,
                            **compare(out.logits, expected, atol, rtol),
                            "actual_top1": out.logits.argmax(-1).flatten().tolist(),
                            "reference_top1": expected.argmax(-1).flatten().tolist(),
                            "top1_exact": bool(
                                torch.equal(out.logits.argmax(-1), expected.argmax(-1))
                            ),
                        }
                    )
                    valid = mask[:, -ids.shape[1] :].bool().cpu()
                    for captured in trace.records[-raw["num_hidden_layers"] :]:
                        layer = captured["layer"]
                        ref = baseline["routing"][f"{layer}:{case['name']}:{step}"]
                        ai, ao = captured["expert_indices"][valid].sort(-1)
                        bi, bo = ref["indices"].sort(-1)
                        report["routing"].append(
                            {
                                "layer": layer,
                                "case": case["name"],
                                "step": step,
                                "routing_exact": bool(
                                    torch.equal(
                                        captured["expert_indices"][valid],
                                        ref["indices"],
                                    )
                                ),
                                "routing_set_exact": bool(torch.equal(ai, bi)),
                                "scores": compare(
                                    captured["routing_weights"][valid].gather(-1, ao),
                                    ref["scores"].gather(-1, bo),
                                    atol,
                                    rtol,
                                ),
                            }
                        )
            trace.close()
    if rank == 0:
        report["numerical_pass"] = all(r["pass"] for r in report["logits"]) and all(
            r["scores"]["pass"] for r in report["routing"]
        )
        report["routing_exact"] = all(r["routing_exact"] for r in report["routing"])
        report["top1_exact"] = all(r["top1_exact"] for r in report["logits"])
        report["bitwise_exact"] = (
            all(r["bitwise_exact"] for r in report["logits"])
            and all(r["scores"]["bitwise_exact"] for r in report["routing"])
            and report["routing_exact"]
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in {"logits", "routing"}}
            ),
            flush=True,
        )
    failed = torch.zeros((), device=device, dtype=torch.int32)
    if rank == 0:
        passed = (
            report["numerical_pass"]
            and report["routing_exact"]
            and report["top1_exact"]
        )
        if "batch_vs_solo_pass" in report:
            passed = passed and report["batch_vs_solo_pass"]
        failed.fill_(int(not passed))
    dist.broadcast(failed, src=0)
    exit_code = int(failed.item())
    dist.destroy_process_group()
    if exit_code:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
