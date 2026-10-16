import json

from moe_serving.datasets import detect_format, load_requests, load_samples


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = "\n".join(message["content"] for message in messages)
        return list(range(len(text.split())))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def test_parsers_keep_the_prompt_and_reference(tmp_path):
    sharegpt = tmp_path / "sharegpt.json"
    sharegpt.write_text(
        json.dumps(
            [
                {
                    "id": "abc",
                    "conversations": [
                        {"from": "human", "value": "one two three four"},
                        {"from": "gpt", "value": "five six"},
                    ],
                },
                {"id": "empty", "conversations": [{"from": "human", "value": "   "}]},
            ]
        ),
        encoding="utf-8",
    )
    parsed = load_requests("sharegpt", sharegpt)
    assert len(parsed) == 1
    assert parsed[0]["source_id"] == "abc"
    assert parsed[0]["messages"] == [{"role": "user", "content": "one two three four"}]
    assert parsed[0]["reference"] == "five six"

    mt = tmp_path / "mt.jsonl"
    mt.write_text(
        json.dumps({"question_id": 81, "category": "writing", "turns": ["draft a note", "revise it"]}) + "\n",
        encoding="utf-8",
    )
    assert load_requests("mt-bench", mt)[0]["messages"][0]["content"] == "draft a note"
    assert detect_format(mt) == "mt-bench"

    gsm = tmp_path / "gsm.jsonl"
    gsm.write_text(json.dumps({"question": "what is two plus two", "answer": "four #### 4"}) + "\n", encoding="utf-8")
    assert load_requests("gsm8k", gsm)[0]["reference"] == "four #### 4"

    humaneval = tmp_path / "he.jsonl"
    humaneval.write_text(
        json.dumps({"task_id": "HumanEval/0", "prompt": "def add ( a , b ) :", "canonical_solution": "return a + b"})
        + "\n",
        encoding="utf-8",
    )
    assert load_requests("humaneval", humaneval)[0]["source_id"] == "HumanEval/0"

    alpaca = tmp_path / "alpaca.json"
    alpaca.write_text(
        json.dumps([{"instruction": "say hello", "input": "Ada", "output": "hello Ada"}]),
        encoding="utf-8",
    )
    alpaca_row = load_requests("alpaca", alpaca)[0]
    assert alpaca_row["messages"][0]["content"] == "say hello\n\nAda"
    assert alpaca_row["reference"] == "hello Ada"
    assert detect_format(alpaca) == "alpaca"


def test_samples_use_the_reference_length_up_to_the_cap(tmp_path):
    path = tmp_path / "messages.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "long",
                        "category": "qa",
                        "messages": [
                            {"role": "user", "content": "word " * 3000},
                            {"role": "assistant", "content": "hidden answer"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "kept",
                        "messages": [
                            {"role": "system", "content": "be brief"},
                            {"role": "user", "content": "one two three four"},
                            {"role": "assistant", "content": "five six seven"},
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    name, samples = load_samples(
        None,
        _Tokenizer(),
        seq_len=20,
        max_new_tokens=None,
        num_samples=None,
        dataset_path=path,
    )
    assert name == "messages"
    assert len(samples) == 1
    assert samples[0]["source_id"] == "kept"
    assert samples[0]["messages"][0]["role"] == "system"
    assert samples[0]["reference"] == "five six seven"
    assert samples[0]["max_new_tokens"] == 3
    assert samples[0]["input_ids"] == [0, 1, 2, 3, 4, 5]

    _name, capped = load_samples(
        "messages",
        _Tokenizer(),
        seq_len=20,
        max_new_tokens=2,
        num_samples=1,
        dataset_path=path,
    )
    assert capped[0]["max_new_tokens"] == 2


def test_serving_mix_stays_available():
    _name, samples = load_samples(
        "serving",
        _Tokenizer(),
        seq_len=2048,
        max_new_tokens=None,
        num_samples=2,
    )
    assert len(samples) == 2
    assert samples[0]["category"] == "short_qa"
    assert samples[0]["input_ids"]
