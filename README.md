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
Q/K/V 持久权重按输出 head 切分，attention sinks 同步切分，O projection 权重按输入列切分。
默认 `--parallel-numerics reference` 保持原生 GEMM 形状、attention 运算形状和专家累加顺序，
用于正确性和 routing 复现；下文说明它的通信及计算代价。
所有 rank 处理相同 token；各 rank 仅保存并计算自己负责的完整专家。
此布局使用 replicated-token EP，不按 token 进行 all-to-all 分发。可分别设置 `--tp-size` / `--ep-size` 为进程数的正整数因子；
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

## Qwen3 MoE 30B / 235B FP8：attention TP + MoE EP

入口 `moe_serving.record_qwen3` 使用本地 `transformers-5.20.0/src` 的
`Qwen3MoeForCausalLM`。实现位于 `moe_serving/qwen3_parallel.py`，无需修改
`transformers-5.20.0/` 或 `LLMs/` 中的文件。

```bash
cd /home/sky/MoE_Serving_Profile
/home/sky/titan-venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m moe_serving.record_qwen3 \
  --checkpoint LLMs/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --tp-size 4 --ep-size 4 --expert-storage fp8 \
  --prompt '请解释 MoE 模型的专家路由。' \
  --prompt 'What is 1+1? Answer briefly.' \
  --max-new-tokens 32 --output outputs/qwen3_30b_routing
```

235B 将 checkpoint 改为 `LLMs/Qwen3-235B-A22B-Instruct-2507-FP8`，
并选择不同的输出目录。两种模型均有 4 个 KV heads，本实现的 TP 大小需整除
query/KV head 数，因而可设为 1、2、4；EP 大小需整除 128 个专家，
TP/EP 大小还都需整除进程数。TP 和 EP 共用同一组 GPU，与 GPT-OSS 使用同样的
连续 rank 子组及 replicated-token EP；默认 reference 模式按原生全局专家顺序合并贡献。
各 rank 只读取本地 attention 分片和本地专家，不在单 rank 装载完整模型。

重复 `--prompt` 即 batched inference，支持不同长度 prompt 的左侧 padding、
greedy decoding、逐 sample EOS 停止和 head-local KV cache。
`--raw-prompt`、`--dtype`、`--max-new-tokens`、`--device` 与 GPT-OSS 一致。
输出同样是 `workload.json` 和每个 sample 的 `000/input.json`、`000/output.json`、
`000/layers/00.json` 等文件；routing 保留全局专家编号，排除 padding 和结束后的 token。

默认 `--expert-storage fp8` 保留 checkpoint 的 block-scaled E4M3 专家权重，
仅在选中某专家时临时反量化成计算 dtype；attention 投影在加载时反量化为计算 dtype，
按 head 切分，并正确处理 TP 切点不与量化 block 对齐的情况。
`--expert-storage bf16` 在加载时反量化本地专家，使用 `--dtype` 指定的计算精度，
会显著增加显存占用。这是参考推理路径，使用普通 linear 运算，未融合 FP8 GEMM，
不模拟 checkpoint 的动态 FP8 activation quantization，性能与数值均以此计算路径为准。
235B 的完整推理需为各 rank 的权重、KV cache 和临时反量化缓冲准备足够显存。

小模型回归包含原生完整模型与 TP/EP 的 logits、routing、KV cache decode、
单条/批量生成一致性，以及真实 FP8 safetensors 和逐 sample JSON：

```bash
/home/sky/titan-venv/bin/python -m pytest \
  tests/test_qwen3_parallel.py tests/test_gpt_oss_parallel.py tests/test_routing_json.py -q
```

## 正确性模式与 4 路验证

GPT-OSS 与 Qwen3 MoE 入口默认使用 `--parallel-numerics reference`，计算精度仍为
BF16，不需要切换 FP32 来规避原有偏差。逐 sample JSON 输出记录 `parallel_numerics`。

修复了四类数值顺序问题：

- TP 局部 GEMM 舍入后再归约，与原生完整 GEMM 不同；分片宽度也会改变 BF16 kernel 的选择。
- attention 的 head 分片改变 batched GEMM 的 batch 维度，导致少数 BF16 舍入差异。
- EP 局部和归约改变专家累加顺序；原生专家输入按 top-k slot、token 排列，旧实现顺序不同。
- Qwen 原生 gate/up 为一次融合投影；GPT-OSS 解包后的矩阵也需保持与原生相同的连续布局。

reference 模式的持久投影权重、专家权重、KV cache 仍分片保存。执行时每次临时汇集
**一个投影矩阵**，由 TP 组 leader 按原生完整形状计算后分发输出；Qwen head norm 和
完整 attention 核也按原生形状在组 leader 执行，再分发 head 输出。
每个专家仍由其所属 EP rank 计算，贡献按全局专家编号顺序广播并累加。
不会在各 rank 持久复制完整模型或完整层的专家权重。

该模式增加权重/激活通信，attention 与投影计算不再获得传统 head 并行的全部加速。
临时 attention 显存按原生全 head 矩阵估算，长 prompt 需为组 leader 留出空间。
`--parallel-numerics fast` 保留 head-local attention 和局部和归约，适合接受数值差异的
性能实验，不提供严格 routing/原生输出一致性承诺。

本机 2026-10-24、4 张 H100、TP=EP=4、BF16 的修复后结果：

| Checkpoint | 层数 | 与原生逐元素一致 | logits 最大绝对误差 | 完整驻留模型复核 |
| --- | ---: | --- | ---: | --- |
| GPT-OSS 20B | 24 | 通过 | 0 | 通过 |
| GPT-OSS 120B | 36 | 通过 | 0 | 通过 |
| Qwen3 MoE 30B FP8 | 48 | 通过 | 0 | 通过 |
| Qwen3 MoE 235B FP8 | 94 | 通过 | 0 | 当前空闲显存不足，完成全 94 层流式对照 |

全 202 层、1212 个逐层 routing 检查均无差异，隐藏状态、routing 分数、各 rank KV cache、
最终 logits 和 top-1 均与原生一致。Qwen3 30B 原失败位置已恢复参考 token `271`，
完整驻留入口复核同样通过。另有 27 项模型/记录回归、7 项验收规则测试通过，
覆盖 FP32/BF16、小模型单进程/2 rank/4 rank、quantized/展开权重、非零 bias、
sliding-window decode、混合 TP/EP、greedy generation、EOS 和 JSON 拆分。
真实 Qwen3 30B 的 4 卡双 sample 命令行及逐 sample JSON 也已验证。

正确性基准是**相同 checkpoint、精度、输入及 batch 布局下匹配原生 eager**。
原生 BF16 自身可能有单条/批量差异，reference 应复现同样的差异。
报告保留 `batch_vs_solo_pass` 作为诊断，用 `batch_semantics_exact` 检查与原生批量语义一致；
reference 模式验收还要求 `bitwise_exact`，不会只凭宽松容差或相同 top-1 判定通过。

真实模型用例为两个不同长度的短 prompt：左填充 batch 和第一个 prompt 的单条请求，
各执行 prefill 与两步固定 token 的 teacher-forced decode。
为在当前空闲显存内验证大模型，参考和并行模型按层流式加载真实权重，所有层均参与
端到端计算，保留完整的层间状态及各层 prefill/decode 因果关系；原生参考不切分 heads 或专家。
可容纳的三个模型另由生产加载器完整驻留复核。235B 不将流式结果描述为完整驻留验证。

范围限于本次输入和环境，不是所有 prompt、长上下文或其他硬件/kernel 的形式化保证。
参考和被测路径使用同精度的 eager 反量化计算，不包含融合 FP8/MXFP4 GEMM 或动态 FP8 activation quantization。

```bash
/home/sky/titan-venv/bin/python -m pytest \
  tests/test_gpt_oss_parallel.py tests/test_qwen3_parallel.py \
  tests/test_routing_json.py tests/test_engram_record.py tests/test_validation_criteria.py -q

/home/sky/titan-venv/bin/python -u tests/run_real_parallel_validation.py \
  --dtypes bfloat16 --parallel-numerics reference --resident \
  --output outputs/correctness_reference_v2 \
  --summary tests/validation_results/four_way_reference.json
```

可增加 `--skip-complete` 复用完整报告；修改实现或输入后应移除该参数并使用新目录。
逐层 JSON、日志和原生参考张量保存在指定 `outputs/` 目录，精简结果为可提交的
`tests/validation_results/four_way_reference.json`，含源码和 checkpoint 配置/索引 SHA256。
修复前的失败证据保留在 `tests/validation_results/four_way.json`。

## GPT-OSS-120B 跑 GSM8K

启动脚本位于 `scripts/run_gpt_oss_120b_gsm8k.sh`，从任意工作目录运行均可。
默认读取本地 `LLMs/gpt-oss-120b`，使用 4 张 GPU、TP=4、EP=4（共用同一组卡）、
MXFP4 专家存储、BF16 计算和 `efficient` 高效模式。
启动脚本默认运行全量测试集，batch size 4，reasoning effort 为 high，每题最多生成 8192 token。
生成预算独立于参考答案长度，不将答案提供给模型，也不按答案长度提前截断。

```bash
bash /home/sky/MoE_Serving_Profile/scripts/run_gpt_oss_120b_gsm8k.sh

# 全量测试集；可用 --start-index 选择起点
bash /home/sky/MoE_Serving_Profile/scripts/run_gpt_oss_120b_gsm8k.sh --num-samples -1

# 快速检查少量问题，或调整批量与输出长度
bash /home/sky/MoE_Serving_Profile/scripts/run_gpt_oss_120b_gsm8k.sh \
  --num-samples 2 --batch-size 2 --max-new-tokens 128

# 只显示 torchrun 命令，不下载、不加载模型、不推理
bash /home/sky/MoE_Serving_Profile/scripts/run_gpt_oss_120b_gsm8k.sh --dry-run
```

GSM8K 默认使用官方测试集 JSONL，首次自动缓存到 `datasets/gsm8k/test.jsonl`。
离线可传 `--dataset-path /path/to/test.jsonl`；`--prepare-only` 只下载/读取数据并按 checkpoint
chat template 分词，生成 `prepared.json`，不加载模型或执行 GPU 推理。
脚本默认使用 `/home/sky/titan-venv/bin/python`，可通过 `PYTHON_BIN` 覆盖。
可设置 `CUDA_VISIBLE_DEVICES=0,1,2,3` 选择参与的卡。

每次启动默认使用新的 `outputs/gpt_oss_120b_gsm8k_<UTC时间>_<PID>/` 目录。
可用 `--output /path/to/run` 或 `GSM8K_OUTPUT` 指定输出；已有运行结果的目录会被拒绝覆盖。

- `run_config.json`：数据集、并行参数、精度、生成预算等配置。
- `workload.json`：运行状态、已完成数量和 sample 目录清单，每批更新。
- `results.jsonl`：问题、参考答案、生成文本、原始 Harmony 控制标记文本与是否达到长度上限；此文件不声称官方 GSM8K 分数。
- `000000/input.json`、`000000/output.json`、`000000/layers/00.json` 等：每个 sample 的输入、输出和每层 prefill/decode routing。

问题按原始顺序选择，参考答案仅写入记录。遇到 EOS 的 sample 独立结束；每批记录写出后释放 recorder。
`reached_token_limit=true` 表示输出可能尚未完成，可提高 `--max-new-tokens` 后在新目录重跑。
reference 路径重视与原生 eager 的数值一致性，其通信和运算开销不代表优化后的服务性能。

本地数据准备与记录逻辑测试：

```bash
/home/sky/titan-venv/bin/python -m pytest \
  tests/test_gsm8k_runner.py tests/test_routing_json.py tests/test_datasets.py -q
```


## 高效 GPU 推理

两个录制入口及 GSM8K 脚本默认 `--parallel-numerics efficient`。
GPT-OSS 20B/120B 使用 FlashAttention 3（保留 attention sink 和滑动窗口）以及
Triton `matmul_ogs`：权重加载时一次性重排 MXFP4，每层两次 grouped GEMM，融合
SwiGLU，不在 forward 内逐专家反量化。Qwen3 30B/235B FP8 使用同样的 FlashAttention 3，
专家使用 block-scaled grouped FP8 GEMM 和动态 FP8 激活。TP 保持本地 heads；EP
计算本地专家后每层一次 all-reduce，不再逐专家广播。TP=4、EP=4 共用 4 张卡。

所有新增实现位于 `moe_serving/`，不修改 `transformers-5.20.0/` 或 `LLMs/`。
依赖本地 Transformers 5.20.0、兼容当前 Torch/CUDA 的 `flash_attn_interface`（FA3）及
`requirements-efficient.txt`。本机 FA3 已安装；其他环境需先安装与硬件匹配的 FA3。

```bash
uv pip install --python /home/sky/titan-venv/bin/python -r requirements-efficient.txt
bash scripts/prepare_efficient_kernels.sh
bash scripts/run_gpt_oss_120b_gsm8k.sh
```

准备脚本下载固定 revision 的内核。Qwen 内核的 Triton 标量 scale 索引兼容补丁在
`~/.cache/moe_serving/` 的私有副本中应用，先修补再导入；推理缺少依赖时明确报错，
不会静默退回 Python 专家循环。`run_config.json` / manifest 的 `backends` 记录后端。
Routing 仍从原生 router 采集全局专家 ID、权重，按 sample 保存输入、输出和逐层 JSON。
GPU routing 数据使用 pinned-memory 异步复制，写 JSON 前等待复制完成。

`reference` 保留用于数值诊断；上文逐位一致的四模型报告仅对应 reference。
高效内核的融合运算、TP 归约顺序及 Qwen FP8 激活量化会改变舍入结果，不能将该报告
套用到 efficient，也不能据短 smoke test 宣称完整 GSM8K 准确率已验证。

可复现的 GPU 检查：

```bash
/home/sky/titan-venv/bin/python tests/helpers/check_flash_attention.py
/home/sky/titan-venv/bin/python tests/helpers/check_grouped_mxfp4.py
/home/sky/titan-venv/bin/python tests/helpers/check_grouped_fp8.py
/home/sky/titan-venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.helpers.check_grouped_ep
/home/sky/titan-venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.helpers.check_efficient_model_parallel
/home/sky/titan-venv/bin/python -m tests.helpers.benchmark_grouped
```

检查包括量化单层误差、4 路 EP 合并，以及使用真实 GPU 内核的小模型 TP1/EP1 对
TP4/EP4、左填充 batch 和越过滑动窗口的 KV decode。精简结果位于
`tests/validation_results/efficient_*.json`。单层基准不含 TP/EP 通信，也不代表端到端加速比。


GSM8K 实时进度默认每 10 秒输出一次，可用 `--log-interval 5` 调整。
日志包括加载阶段、每批 sample ID / prompt 长度、prefill 后第一步、生成步数、已结束样本、
累计有效生成 token、batch 总吞吐、JSON 写出状态和整体完成比例 / ETA。
日志同步保存到运行目录的 `progress.log`，可在另一个终端运行
`tail -f outputs/<运行目录>/progress.log`。首次 prefill 含内核编译，第一步可能较慢；
ETA 按已保存样本的平均耗时估算，生成长度不同会使估计波动。


每个 GSM8K batch 推理结束后，同步写出该批全部 sample 的 `input.json`（文本 / token）、
`output.json`（生成文本 / token）和 `layers/*.json`（每层 MoE 的 routing choice / score）。
JSON 采用临时文件、文件 `fsync` 和原子替换；`results.jsonl`、目录项也显式同步。
最后写入 `batches/<首sample ID>-<末sample ID>.json` 完成标记。
只有写出成功并通知全部 rank 后才开始下一批；失败会终止运行，不会继续吞掉写入错误。
终端的 `Batch ... saved` 表示本批同步写出已完成。这里不保存 MoE hidden-state 张量。
