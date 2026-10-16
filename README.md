# MoE Serving Profile

用本地 checkpoint 记录 MoE routing。Qwen3.8-Flash-Next FP8 走仓库里的 Transformers 5.19.0 源码，不走环境里另外装的那一份。

源码放在 `third_party/transformers-5.19.0/src`（PyPI `transformers==5.19.0` 的 sdist），不提交进仓库。`moe_serving.record_routing` 会把它插到 `sys.path` 最前面，然后调用 `Qwen4ExpForConditionalGeneration`。下载方式见 `third_party/README.md`。路由来自其中的 `Qwen4ExpTextTopKRouter.forward`，返回 logits、归一化后的 top-k 分数和专家编号。

默认 checkpoint：`/home/sky/MoE_Serving_Profile/LLMs/Qwen3.8-Flash-Next-FP8`

```bash
source /home/sky/llm-venv/bin/activate
python -m moe_serving.record_routing \
    --checkpoint /home/sky/MoE_Serving_Profile/LLMs/Qwen3.8-Flash-Next-FP8 \
    --max-new-tokens 16 \
    --output outputs/qwen38_fp8_routing.pt
```

有 GPU 时按 `device_map=auto` 装到各卡。专家权重是 block size 128 的 FP8，计算走 `grouped_mm`。`model.generate` 先做一次 prefill，再按 token decode，每一步的专家编号都写进 `--output`。`expert_load_cv` 越小越均衡。环境用 `/home/sky/llm-venv`，不要用 `skypilot-runtime`。

## 另一条专家并行运行时

`python -m moe_serving.cli profile` 是仓库自己的专家并行实现，读 BF16 fused 专家，Engram 表 mmap 在 DRAM。它不加载这份 FP8 checkpoint。Qwen3 MoE 和 Qwen3.6 用 `--model qwen3_moe` 或 `--model qwen3_6`。

```bash
PYTHONPATH=. pytest tests
```
