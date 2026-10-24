# MoE Serving Profile

用本地 checkpoint 记录 MoE routing。Qwen3.8-Flash-Next FP8 走仓库里的 Transformers 5.20.0 源码，不走环境里另外装的那一份。

源码放在 `transformers-5.20.0/src`（PyPI `transformers==5.20.0` 的 sdist），不提交进仓库。`moe_serving.record_routing` 会把它插到 `sys.path` 最前面，然后调用 `Qwen4ExpForConditionalGeneration`。路由来自其中的 `Qwen4ExpTextTopKRouter.forward`，返回 logits、归一化后的 top-k 分数和专家编号。

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

## GPT-OSS-20B / 120B：attention TP + MoE EP

新增入口 `moe_serving.record_gpt_oss`，固定使用项目根目录的
`transformers-5.20.0/src`。不需要修改该源码树，也不会加载环境里的另一份
Transformers。运行环境需要与这份源码匹配的依赖、PyTorch、safetensors、tokenizers；
当前机器已在 `/home/sky/titan-venv` 验证。Qwen 和 GPT-OSS 入口均使用这份本地 5.20.0 源码。

```bash
cd /home/sky/MoE_Serving_Profile
/home/sky/titan-venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m moe_serving.record_gpt_oss \
  --checkpoint LLMs/gpt-oss-20b \
  --tp-size 4 --ep-size 4 \
  --prompt '请解释 MoE 模型的专家路由。' \
  --max-new-tokens 32 --output outputs/gpt_oss_20b_routing
```

120B 使用同一命令，把 checkpoint 改为 `LLMs/gpt-oss-120b`，输出路径改为
`outputs/gpt_oss_120b_routing`。默认应用 checkpoint 的 chat template；
`--raw-prompt` 可直接输入原始文本。重复 `--prompt` 支持左侧 padding 的批量输入。
生成采用 greedy decoding，并使用原生 KV cache；每张卡只缓存自己负责的 KV heads。

TP 和 EP 默认都等于进程数，**共用这些 GPU，无需 TP×EP 张卡**。
Q/K/V 按输出 head 切分，attention sinks 同步切分，O projection 按输入列切分，
all-reduce 后只加一次输出 bias。所有 rank 处理相同 token；各 rank 仅保存并计算
自己负责的完整专家，MoE 输出通过 all-reduce 合并。此布局不是按 token 分发的
all-to-all EP。可分别设置 `--tp-size` / `--ep-size` 为进程数的正整数因子；
使用连续 rank 子组。TP 必须同时整除 query heads 和 KV heads，EP 必须整除专家数。
若两个并行度均小于进程数，存在重复计算，通常应使较大者等于进程数。
仅支持同一节点的同一批输入，不提供多请求调度或数据并行。

加载器在 meta device 建模，通过 safetensors 切片直接读取本 rank 的权重。
默认 `--expert-storage mxfp4` 保留原始 MXFP4 blocks/scales，每次计算选中的专家时
使用本地 Transformers 的解码函数临时展开为 BF16；不依赖下载 Triton kernel。
这是一条显存友好的参考推理路径，**没有融合 MXFP4 GEMM，速度不代表优化后的服务性能**。
`--expert-storage bf16` 可以在加载时展开本 rank 的专家，减少反复解码，但 120B
需要按 BF16 全部专家估算显存，不能按 61GB 的压缩文件大小估算。
Embedding、router、norm 和 LM head 复制在各 rank 上。

只有 rank 0 写 JSON；单条请求和 batched inference 使用相同结构，每个 sample 一个目录：

```text
outputs/gpt_oss_20b_routing/
  workload.json
  000/
    input.json
    output.json
    layers/
      00.json
      01.json
  001/
    ...
```

`input.json` 保存 prompt 和输入 token；`output.json` 保存生成文本、token 和并行配置。
每层 JSON 的 `steps` 包含 prefill/decode 的 token、位置、专家编号 `indices` 和分数 `scores`。
批量记录按 sample 拆分，移除左侧 padding 和结束后的无效 token，保留 EOS。
Qwen 记录也始终放在 sample 子目录中，另有 `engram/` 保存对应层的表行号。
旧的 `.pt` / `.json` 输出参数会去掉后缀，作为目录使用。

路由属于本次送入模型的 token；最后一个新生成 token 若未再次送入模型，则没有其路由。
路由张量实时搬到 CPU，记录会增加同步开销，长上下文/长输出会增加主机内存占用。

小模型回归测试（独立子进程，避免与 5.20.0 导入冲突）：

```bash
/home/sky/titan-venv/bin/python -m pytest tests/test_gpt_oss_parallel.py -q
```

测试覆盖单进程、两进程 TP+EP、非零 bias、MXFP4/展开权重、批量 padding、
跨 sliding window 的 KV cache decode、逐层 routing 和生成记录。
