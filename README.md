# deck

deck is a zero-shot classification framework. You write a taxonomy, deck helps
you describe the classes, and it sends easy examples to a small CPU model and
hard examples to a teacher LLM. Decisions are scored by **deck4b**.

## Models

| id | role | runtime |
|---|---|---|
| `deck4b` | decide | GPU, FP8 weight-only |
| `potion` | label easy examples | CPU |
| `teacher` | write class descriptions and label hard examples | LLM |

`deck models` prints this registry. Potion weights are not in this repository
yet. The routing policy is: short, ordinary examples go to potion; long text,
exceptions, dates, and large option sets go to the teacher. deck4b remains the
decision model either way.

## Taxonomy

A taxonomy is a JSON file: an id, a question, and classes with descriptions.

```bash
python -m deck taxonomy examples/support.json
python -m deck describe examples/support.json
python -m deck describe examples/support.json --apply teacher.json
python -m deck route examples/support.json --state "Where is my order?"
python -m deck ui examples/support.json
```

`deck ui` opens a page where you can edit class descriptions and see whether
an example would be labeled by potion or the teacher.

`deck describe` prints a prompt for the teacher. The teacher must return JSON
that keeps every class id and adds `description`, `exclusions`, and one
synthetic `example`. `--apply` writes a parsed response back into the taxonomy
and ignores unknown ids.

## deck4b

deck4b is a rank-8 LoRA. We trained the adapter. We did not train the base
checkpoint.

- Served id: `deck4b`.
- Weights: [`krishna765/deck-4b-v1.0`](https://huggingface.co/krishna765/deck-4b-v1.0)
  at `81e9f5e2cc701f86b1be700dfb49f6549ad83f39`, Apache-2.0.
- Base checkpoint: `alibiserikbay/JevK5` at
  `c4f7fdb3aeab5582336406e78d3bef11bf98833d` (Apache-2.0), itself built on
  Qwen3.5-4B (Apache-2.0). See `NOTICE`.
- Runtime: the LoRA is merged, then linear weights are quantized to FP8
  weight-only with TorchAO. The JevK5 runtime
  (`allebee/jevk5@f944fe37ff1d5ed3830aa4c8d88b7189c8c1268a`) provides the
  prompt, tokenization, and HTTP protocol.
- Readout: calibrated softmax over A-P next-token logits; no generation.
- Temperature: `1.159999966621399`, read from the adapter's `joint_config.json`.

Needs an FP8-capable NVIDIA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
python -m pip install -e '.[serve]'

deck serve --host 0.0.0.0 --port 8090
curl --fail http://127.0.0.1:8090/health
```

The adapter repository and revision are the defaults. `--no-graphs` disables
CUDA graphs.

### JevBench public reference

Against JevBench revision `1bcc55eb6c8cffde2306b3db03ede39b61c6152a`, official
`typesafe` adapter, loopback, one request at a time. Two serial runs returned
identical probabilities on every item: 231/231 valid, 0 failures.

| Tier | Accuracy | ECE10 | p50 | p95 | Mean input tokens |
|---|---:|---:|---:|---:|---:|
| Easy | 48/48 | 0.017 | 50.0 ms | 50.5 ms | 163.8 |
| Original | 68/72 | 0.051 | 50.0 ms | 50.6 ms | 168.0 |
| Hard | 87/111 | 0.041 | 60.5 ms | 141.8 ms | 1,274.1 |
| Overall | 203/231 | 0.023 | 50.6 ms | 116.9 ms | — |

Overall Brier score was `0.18351`. These are local H100 loopback numbers, not
an official leaderboard row. The summary is `results/fp8_validation.json`.

No JevBench row was used in gradient training or temperature fitting. The 231
public items were used to choose this adapter and the FP8 runtime, so the
public numbers are development-directed. An eight-word and exact-state audit
of the 1,000-row candidate pool found zero hits
(`results/contamination_audit.json`).

```bash
python scripts/contamination_audit.py \
  --train data/training_mix.jsonl \
  --jevbench-root /path/to/jevbench/datasets/public
```

The training file is not in this repository because its source licenses
differ. Its SHA-256 is
`9ace459104d932f05a81474f07a37053f50a0d88ad12d5830ea826abb8c4bca4`.

Limits: English only. Up to 16 options in one pass; larger sets use knockout
passes. Inputs above 16,384 tokens are rejected, never truncated.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
