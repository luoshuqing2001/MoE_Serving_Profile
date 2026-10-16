"""Chat requests shaped like a serving trace.

vLLM and SGLang replay ShareGPT conversations for this. Azure's public LLM
traces and BurstGPT do the same with production arrival times and input/output
lengths: many short questions, some code and summarization, a few longer
multi-turn chats. This module is that mix, tokenized with the checkpoint chat
template, so each request can be recorded on its own.
"""

from __future__ import annotations

REQUESTS: list[dict] = [
    {
        "category": "short_qa",
        "max_new_tokens": 64,
        "messages": [
            {"role": "user", "content": "巴黎是哪个国家的首都？只回答国家名。"},
        ],
    },
    {
        "category": "explanation",
        "max_new_tokens": 128,
        "messages": [
            {"role": "user", "content": "用三句话解释专家并行和数据并行的差别。"},
        ],
    },
    {
        "category": "translation",
        "max_new_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": "把这句话翻译成英文：专家并行把不同的专家放到不同的 GPU 上，token 再路由过去。",
            },
        ],
    },
    {
        "category": "code",
        "max_new_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": "用 Python 写一个函数，输入整数列表，返回去重后仍保持原顺序的新列表。只给函数。",
            },
        ],
    },
    {
        "category": "summarize",
        "max_new_tokens": 96,
        "messages": [
            {
                "role": "user",
                "content": (
                    "用两句话概括下面这段话。\n"
                    "线上推理和训练时的数据并行不一样。服务系统里，请求一道一道到达，"
                    "prompt 有的只有几十个 token，有的把整篇文档贴进来。输出长度也差很多："
                    "有的请求只回一个词，有的要写完一整段解释。调度器要把正在 prefill 的长请求"
                    "和正在 decode 的短请求放在一起，否则 GPU 会一会儿空、一会儿被一个长 prompt 占满。"
                    "对 MoE 来说，每个 token 还会再选一批专家，所以同一批请求里不同 token 打到的专家也不一样。"
                ),
            },
        ],
    },
    {
        "category": "multi_turn",
        "max_new_tokens": 128,
        "messages": [
            {"role": "user", "content": "什么是 Mixture of Experts？"},
            {
                "role": "assistant",
                "content": "Mixture of Experts 是一层里放很多专家网络，每个 token 只送给其中少数几个。",
            },
            {"role": "user", "content": "那和把整个模型复制到每张卡上有什么不同？请用两句话说明。"},
        ],
    },
    {
        "category": "rewrite",
        "max_new_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": "把这句话改得更正式：这个模型跑 serving 的时候，长短请求混在一起，别只测一条。",
            },
        ],
    },
    {
        "category": "math",
        "max_new_tokens": 96,
        "messages": [
            {
                "role": "user",
                "content": "一批里有 4 个请求，每个请求 8 个 token，top-8 专家。一共会发生多少次专家分配？给出算式和结果。",
            },
        ],
    },
]


def render_messages(tokenizer, messages: list[dict], seq_len: int, truncate: bool = True) -> list[int]:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(encoded, "input_ids"):
        token_ids = encoded["input_ids"]
    else:
        token_ids = encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    if truncate and seq_len and len(token_ids) > seq_len:
        token_ids = token_ids[:seq_len]
    return token_ids


def serving_samples(tokenizer, seq_len: int) -> list[dict]:
    """Return one record per serving request, with chat-template token ids."""
    samples = []
    for index, request in enumerate(REQUESTS):
        token_ids = render_messages(tokenizer, request["messages"], seq_len)
        user_text = request["messages"][-1]["content"]
        samples.append(
            {
                "id": f"{index:03d}",
                "category": request["category"],
                "max_new_tokens": request["max_new_tokens"],
                "user_text": user_text,
                "messages": request["messages"],
                "input_ids": token_ids,
            }
        )
    return samples
