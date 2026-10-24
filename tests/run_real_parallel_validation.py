"""Reproduce four-checkpoint TP=EP=4 comparisons and preserve successes/failures.

Run from the repository root with the same Python environment as inference.
Raw reports and native baselines go to outputs/. A compact, reviewable summary
is saved outside ignored model/source/output directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "gpt_oss_20b": "gpt-oss-20b",
    "gpt_oss_120b": "gpt-oss-120b",
    "qwen3_30b": "Qwen3-30B-A3B-Instruct-2507-FP8",
    "qwen3_235b": "Qwen3-235B-A22B-Instruct-2507-FP8",
}


def passed(report):
    native = all(
        report.get(k, False) for k in ("numerical_pass", "routing_exact", "top1_exact")
    )
    batch = report.get("batch_semantics_exact", report.get("batch_vs_solo_pass", True))
    exact = True
    if report.get("parallel_numerics") == "reference":
        exact = report.get(
            "bitwise_exact",
            bool(report.get("logits"))
            and bool(report.get("routing"))
            and all(r.get("bitwise_exact", False) for r in report.get("logits", []))
            and all(
                r["scores"].get("bitwise_exact", False)
                for r in report.get("routing", [])
            ),
        )
    return native and batch and exact


def run(script, arguments, log):
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(ROOT / "tests/helpers" / script),
        *map(str, arguments),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        output.write(" ".join(cmd) + "\n")
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=str(ROOT)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            output.write(line)
            output.flush()
            if line.startswith("{"):
                message = json.loads(line)
                if "layer" not in message or (message["layer"] + 1) % 10 == 0:
                    print(line.rstrip(), flush=True)
        return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS)
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "bfloat16"],
        default=["float32", "bfloat16"],
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/correctness_reference"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "tests/validation_results/four_way_reference.json",
    )
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument(
        "--parallel-numerics", choices=["reference", "fast"], default="reference"
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="Also check the fully resident production loader where memory permits",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.summary = args.summary.resolve()
    summary = {
        "tp_size": 4,
        "ep_size": 4,
        "cases": [],
        "source_sha256": {},
        "scope": "Two short prompts, left-padded batch plus solo, prefill and two teacher-forced decode steps. "
        "Native eager reference with matching dequantized compute dtype; no fused quantized activation kernels.",
    }
    import torch

    summary["torch_version"] = torch.__version__
    summary["gpu"] = torch.cuda.get_device_name(0)
    for relative in [
        "moe_serving/gpt_oss_parallel.py",
        "moe_serving/parallel_numerics.py",
        "moe_serving/qwen3_parallel.py",
        "tests/helpers/real_parallel_worker.py",
        "tests/helpers/resident_parallel_worker.py",
        "transformers-5.20.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py",
        "transformers-5.20.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
    ]:
        summary["source_sha256"][relative] = hashlib.sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()
    for model in args.models:
        checkpoint = ROOT / "LLMs" / MODELS[model]
        for dtype in args.dtypes:
            stem = model + ("_fp32" if dtype == "float32" else "_bf16")
            path = args.output / (stem + ".json")
            report = json.loads(path.read_text()) if path.exists() else {}
            can_reside = args.resident and model != "qwen3_235b"
            reference = path.with_suffix(".reference.pt")
            if not (
                args.skip_complete
                and report.get("status") == "complete"
                and report.get("parallel_numerics") == args.parallel_numerics
                and (not can_reside or reference.exists())
            ):
                print(f"Streamed native comparison: {model} {dtype}", flush=True)
                code = run(
                    "real_parallel_worker.py",
                    [
                        "--checkpoint",
                        checkpoint,
                        "--dtype",
                        dtype,
                        "--report",
                        path,
                        "--parallel-numerics",
                        args.parallel_numerics,
                    ],
                    path.with_suffix(".log"),
                )
                report = json.loads(path.read_text()) if path.exists() else {}
                if report.get("status") != "complete":
                    raise RuntimeError(
                        f"Validation did not complete: {path}, exit {code}"
                    )
            expected_layers = json.loads((checkpoint / "config.json").read_text())[
                "num_hidden_layers"
            ]
            if not report.get("full_model") or len(report["layers"]) != expected_layers:
                raise RuntimeError(f"A full checkpoint comparison is required: {path}")
            # Earlier completed bitwise reports already prove the same equality:
            # matching native tensors in both cases imply matching batch deltas.
            if report.get("bitwise_exact") and "batch_semantics_exact" not in report:
                report["batch_semantics_exact"] = True
                report["batch_semantics_evidence"] = (
                    "Derived from bitwise equality to native in every batch/solo case"
                )
            result = {
                k: report[k]
                for k in [
                    "dtype",
                    "parallel_numerics",
                    "atol",
                    "rtol",
                    "numerical_pass",
                    "routing_exact",
                    "top1_exact",
                    "batch_vs_solo_pass",
                    "seconds",
                    "decode_steps",
                    "token_rows",
                ]
            }
            result.update(
                model=model,
                layers=len(report["layers"]),
                full_model=report["full_model"],
                passed=passed(report),
                report=str(path.relative_to(ROOT)),
            )
            result["max_logit_abs_error"] = max(r["max_abs"] for r in report["logits"])
            result["max_logit_relative_l2"] = max(
                r["relative_l2"] for r in report["logits"]
            )
            result["logits_within_tolerance"] = all(r["pass"] for r in report["logits"])
            result["bitwise_exact"] = report.get("bitwise_exact", False)
            result["batch_semantics_exact"] = report.get("batch_semantics_exact", False)
            result["top1_mismatches"] = [
                {k: r[k] for k in ("case", "step", "actual_top1", "reference_top1")}
                for r in report["logits"]
                if not r["top1_exact"]
            ]
            result["hidden_checks_differing"] = sum(
                not r["hidden"]["pass"]
                for layer in report["layers"]
                for r in layer["results"]
            )
            result["cache_within_tolerance"] = all(
                m["pass"]
                for layer in report["layers"]
                for r in layer["results"]
                for m in r["cache"]
            )
            result["max_hidden_relative_l2"] = max(
                r["hidden"]["relative_l2"]
                for layer in report["layers"]
                for r in layer["results"]
            )
            result["routing_checks"] = sum(
                len(layer["results"]) for layer in report["layers"]
            )
            result["routing_checks_differing"] = sum(
                not r["routing_exact"]
                for layer in report["layers"]
                for r in layer["results"]
            )
            result["batch_vs_solo"] = report["batch_vs_solo"]
            if reference.exists():
                native = torch.load(reference, map_location="cpu", weights_only=True)
                comparisons = []
                for step in range(native["decode_steps"] + 1):
                    batch = native["logits"][f"batch:{step}"][:1]
                    solo = native["logits"][f"solo:{step}"]
                    delta = (batch - solo).abs()
                    comparisons.append(
                        {
                            "step": step,
                            "max_abs": float(delta.max()),
                            "relative_l2": float(
                                (batch - solo).norm() / solo.norm().clamp_min(1e-12)
                            ),
                            "pointwise_pass": bool(
                                (
                                    delta
                                    <= report["atol"] + report["rtol"] * solo.abs()
                                ).all()
                            ),
                            "top1_exact": bool(
                                torch.equal(batch.argmax(-1), solo.argmax(-1))
                            ),
                        }
                    )
                result["native_batch_vs_solo"] = comparisons
                del native
            result["config_sha256"] = hashlib.sha256(
                (checkpoint / "config.json").read_bytes()
            ).hexdigest()
            result["weight_index_sha256"] = hashlib.sha256(
                (checkpoint / "model.safetensors.index.json").read_bytes()
            ).hexdigest()
            if can_reside:
                resident_path = args.output / (stem + "_resident.json")
                if not (args.skip_complete and resident_path.exists()):
                    print(
                        f"Production fully resident comparison: {model} {dtype}",
                        flush=True,
                    )
                    code = run(
                        "resident_parallel_worker.py",
                        ["--reference", reference, "--report", resident_path],
                        resident_path.with_suffix(".log"),
                    )
                    if not resident_path.exists():
                        raise RuntimeError(
                            f"Resident validation did not complete: {resident_path}, exit {code}"
                        )
                resident = json.loads(resident_path.read_text())
                result["resident"] = {
                    k: resident[k]
                    for k in ("numerical_pass", "routing_exact", "top1_exact")
                }
                result["resident"]["bitwise_exact"] = resident.get(
                    "bitwise_exact",
                    all(r.get("bitwise_exact", False) for r in resident["logits"])
                    and all(
                        r["scores"].get("bitwise_exact", False)
                        for r in resident["routing"]
                    )
                    and resident["routing_exact"],
                )
                result["resident"]["report"] = str(resident_path.relative_to(ROOT))
                result["resident"]["top1_mismatches"] = [
                    {
                        k: r.get(k)
                        for k in ("case", "step", "actual_top1", "reference_top1")
                    }
                    for r in resident["logits"]
                    if not r["top1_exact"]
                ]
                result["resident"]["max_logit_abs_error"] = max(
                    r["max_abs"] for r in resident["logits"]
                )
                result["passed"] &= passed(resident)
            elif model == "qwen3_235b":
                result["resident"] = (
                    "Not run: current free GPU memory cannot hold the full checkpoint per rank"
                )
            else:
                result["resident"] = (
                    "Not requested, or streamed comparison failed; see flags above"
                )
            summary["cases"].append(result)
            summary["passed"] = all(r["passed"] for r in summary["cases"])
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
