"""Audit the deck-4B training mix against JevBench public text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(value) -> list[str]:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True)
    return TOKEN.findall(value.lower())


def _ngrams(value, size: int) -> set[tuple[str, ...]]:
    tokens = _tokens(value)
    return {
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def _training_text(row: dict) -> str:
    options = " ".join(
        f"{option['id']} {option.get('description', '')}"
        for option in row["options"]
    )
    return f"{row['state']} {row['question']} {options}"


def _benchmark_text(row: dict) -> str:
    question = row["question"]
    criteria = question.get("criteria")
    return (
        f"{row['state']} {question['instructions']} "
        f"{json.dumps(criteria, sort_keys=True)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/training_mix.jsonl"),
    )
    parser.add_argument(
        "--jevbench-root",
        type=Path,
        required=True,
        help="path to a JevBench checkout's datasets/public directory",
    )
    parser.add_argument("--ngram", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/contamination_audit.json"),
    )
    args = parser.parse_args()

    benchmark = []
    benchmark_ngrams: dict[tuple[str, ...], set[str]] = {}
    for tier in ("easy", "original", "hard"):
        path = args.jevbench_root / f"{tier}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            benchmark.append(row)
            for ngram in _ngrams(_benchmark_text(row), args.ngram):
                benchmark_ngrams.setdefault(ngram, set()).add(str(row["id"]))

    training = [
        json.loads(line)
        for line in args.train.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hits = []
    exact_state_hits = []
    benchmark_states = {
        " ".join(_tokens(row["state"])): str(row["id"]) for row in benchmark
    }
    for row in training:
        normalized_state = " ".join(_tokens(row["state"]))
        if normalized_state in benchmark_states:
            exact_state_hits.append(
                {
                    "train_id": row["id"],
                    "benchmark_id": benchmark_states[normalized_state],
                }
            )
        shared = {}
        for ngram in _ngrams(_training_text(row), args.ngram):
            for benchmark_id in benchmark_ngrams.get(ngram, ()):
                shared.setdefault(benchmark_id, []).append(" ".join(ngram))
        for benchmark_id, sequences in sorted(shared.items()):
            hits.append(
                {
                    "train_id": row["id"],
                    "train_source": row["source"],
                    "benchmark_id": benchmark_id,
                    "shared_sequences": sorted(set(sequences)),
                }
            )

    report = {
        "training_file": str(args.train),
        "training_file_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "training_records": len(training),
        "jevbench_public_records": len(benchmark),
        "ngram_words": args.ngram,
        "exact_normalized_state_hits": exact_state_hits,
        "shared_ngram_hits": hits,
        "passed": not exact_state_hits and not hits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
