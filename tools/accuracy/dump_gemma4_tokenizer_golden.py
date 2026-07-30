"""Dump a Gemma 4 tokenizer/chat-template golden from a local checkpoint.

    python tools/accuracy/dump_gemma4_tokenizer_golden.py <model-dir> <out.json> \
        --source-repo google/gemma-4-12B-it --revision <sha>

Provenance is required rather than inferred: a checkpoint directory carries no
reliable record of where it came from, and guessing from the path would write a
wrong repository name into the golden. Every chat case is expected to render; a
template error aborts the dump rather than being written into the golden.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import transformers
from transformers import AutoTokenizer

PROBES = [
    # Both sides run the same `tokenizers` crate, so these probe the layers that
    # can actually differ: the Python wrapper's added-token and
    # add_special_tokens handling, and version skew in the shared library. One
    # case per algorithm class rather than broad script coverage.
    ("empty", ""),
    ("ascii_words", "Hello, world!"),
    ("multi_space", "a    b\tc\nd\r\ne"),
    ("digits_run", "12345"),
    ("cjk", "\u4f60\u597d\u4e16\u754c"),
    ("emoji_zwj", "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466"),
    ("combining_marks", "e\u0301a\u0300o\u0302"),
    ("control_chars", "a\x00b\x01c\x7f"),
    ("special_bos_literal", "<bos>"),
    ("special_image_literal", "<|image|>"),
    ("special_in_sentence", "before <|image|> after"),
]

SPECIAL_TOKEN_KEYS = [
    "bos_token",
    "eos_token",
    "pad_token",
    "unk_token",
    "mask_token",
    "eot_token",
    "image_token",
    "audio_token",
    "boi_token",
    "eoi_token",
    "boa_token",
    "eoa_token",
    "think_token",
    "escape_token",
]

CHAT_CASES = [
    (
        "single_user_with_generation_prompt",
        [{"role": "user", "content": "What is 2 + 2?"}],
        True,
    ),
    (
        "single_user_no_generation_prompt",
        [{"role": "user", "content": "What is 2 + 2?"}],
        False,
    ),
    (
        "multi_turn",
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there."},
            {"role": "user", "content": "And now?"},
        ],
        True,
    ),
    (
        "system_then_user",
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Explain gravity."},
        ],
        True,
    ),
    (
        "unicode_content",
        [{"role": "user", "content": "翻译：🙂 と こんにちは"}],
        True,
    ),
]

HASHED_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def token_value(raw):
    return raw["content"] if isinstance(raw, dict) else raw


def dump_file_hashes(model_dir: Path) -> dict:
    hashes = {}
    for name in HASHED_FILES:
        path = model_dir / name
        if not path.exists():
            raise SystemExit(f"required file missing from the checkpoint: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def dump_special_tokens(tokenizer, tokenizer_config: dict) -> dict:
    specials = {}
    for key in SPECIAL_TOKEN_KEYS:
        value = token_value(tokenizer_config.get(key))
        if value is None:
            raise SystemExit(
                f"required special token missing from tokenizer_config: {key}"
            )
        specials[key] = {"token": value, "id": tokenizer.convert_tokens_to_ids(value)}
    return specials


def dump_probes(tokenizer) -> list:
    return [
        {
            "name": name,
            "text": text,
            "ids_plain": tokenizer(text, add_special_tokens=False)["input_ids"],
            "ids_with_specials": tokenizer(text, add_special_tokens=True)["input_ids"],
        }
        for name, text in PROBES
    ]


def dump_chat_templates(tokenizer) -> list:
    cases = []
    for name, messages, add_generation_prompt in CHAT_CASES:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        cases.append(
            {
                "name": name,
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "rendered": rendered,
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("out")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())

    golden = {
        "source_repo": args.source_repo,
        "revision": args.revision,
        "transformers_version": transformers.__version__,
        "file_sha256": dump_file_hashes(model_dir),
        "special_tokens": dump_special_tokens(tokenizer, tokenizer_config),
        "probes": dump_probes(tokenizer),
        "chat_templates": dump_chat_templates(tokenizer),
    }

    Path(args.out).write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n")
    print(
        f"wrote {args.out}: {len(golden['probes'])} probes, "
        f"{len(golden['chat_templates'])} chat cases"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
