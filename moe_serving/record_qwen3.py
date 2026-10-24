"""Qwen3 MoE 30B/235B FP8 TP/EP inference and per-sample routing JSON.

Use the same CLI options and output layout as moe_serving.record_gpt_oss.
"""

from moe_serving.record_gpt_oss import main as _main


def main(argv=None):
    _main(argv, model_family="qwen3_moe")


if __name__ == "__main__":
    main()
