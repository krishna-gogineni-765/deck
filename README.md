# deck-4B v1.0

FP8 server for deck-4B v1.0 on JevBench's `typesafe` adapter. It exposes
`POST /v1/systemone` and `GET /health`, supports `noul`, `choice`, and
`score`, returns native probabilities over the exact labels, and emits zero
output tokens.

## Model

- Name: deck-4B, version v1.0. Served id: `deck-4b-v1.0`.
- Adapter: [`krishna765/deck-4b-v1.0`](https://huggingface.co/krishna765/deck-4b-v1.0)
  at `81e9f5e2cc701f86b1be700dfb49f6549ad83f39`, rank-8 LoRA, Apache-2.0.
- Base checkpoint: `alibiserikbay/JevK5` at
  `c4f7fdb3aeab5582336406e78d3bef11bf98833d` (Apache-2.0), itself built on
  Qwen3.5-4B (Apache-2.0). We trained the adapter. We did not train the base
  checkpoint. See `NOTICE`.
- Runtime: the LoRA is merged into the base, then all linear weights are
  quantized to FP8 weight-only with TorchAO. The JevK5 runtime
  (`allebee/jevk5@f944fe37ff1d5ed3830aa4c8d88b7189c8c1268a`) provides the
  prompt, tokenization, and HTTP protocol.
- Readout: calibrated softmax over A-P next-token logits; no generation.
- Temperature: `1.159999966621399`, read at startup from the adapter's
  `joint_config.json`. It was fitted by NLL on the adapter's 65-row
  validation split.

## Start

Use an H100 or another GPU with FP8 support.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
python -m pip install -e .

deck-4b-serve --host 0.0.0.0 --port 8090
curl --fail http://127.0.0.1:8090/health
```

The adapter repository and revision are pinned as defaults. Override them
with `--adapter` and `--adapter-revision`; `--adapter` also accepts a local
folder. `--no-graphs` disables CUDA graphs.

## Official public validation

Against JevBench revision `1bcc55eb6c8cffde2306b3db03ede39b61c6152a`:

```bash
python -m jevbench.cli run \
  --adapter typesafe \
  --endpoint http://127.0.0.1:8090 \
  --model deck-4b-v1.0 \
  --key-env '' \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --results RUN/results.jsonl \
  --raw-dir RUN/raw \
  --ledger RUN/ledger.jsonl \
  --reserve-usd 0 \
  --cost-basis self_hosted_h100_unpriced \
  --manifest RUN/manifest.json \
  --run-label deck-4b-v1.0
```

Two clean serial runs returned identical probabilities for every item, with
231/231 valid responses and no failures:

| Tier | Accuracy | ECE10 | p50 | p95 | Mean input tokens |
|---|---:|---:|---:|---:|---:|
| Easy | 48/48 | 0.017 | 50.0 ms | 50.5 ms | 163.8 |
| Original | 68/72 | 0.051 | 50.0 ms | 50.6 ms | 168.0 |
| Hard | 87/111 | 0.041 | 60.5 ms | 141.8 ms | 1,274.1 |
| Overall | 203/231 | 0.023 | 50.6 ms | 116.9 ms | — |

Overall Brier score was `0.18351`. Latency is loopback caller wall time on one
H100 and will not match JevBench's remote measurement. The summary is in
`results/fp8_validation.json`.

A batch-oriented FP8 experiment reached 205/231, but that is **not** the
submitted result: JevBench measures one request at a time.

## Training

One bf16 epoch, learning rate `1e-5`, LoRA rank 8, alpha 16, dropout 0.05
over Qwen3.5 attention and linear-attention projections. Data: the first 500
records of a seeded, nested 1,000-record candidate mix; a stable hash split
gave 435 training and 65 validation rows. Sources: synthetic tool selection,
SNIPS and Mozilla smart-intent examples, GLiClass/GliClass-RAC records,
Open-Jev synthetic/control records, and deterministic RLCD examples.

The training rows are not redistributed here because their source licenses
differ. The 1,000-record file has SHA-256
`9ace459104d932f05a81474f07a37053f50a0d88ad12d5830ea826abb8c4bca4`.

## Development and contamination disclosure

No JevBench row was used in gradient training or temperature fitting. However,
the 231 public items were repeatedly evaluated during development:

- to compare the base checkpoint with 500-row and 1,000-row adaptation variants;
- to select this 500-row adapter as deck-4B v1.0;
- to compare bf16/INT8/FP8 serving choices and finalize this FP8 runtime.

Therefore the public numbers are development-directed and must not be
presented as untouched test performance. JevBench's sealed evaluation is the
meaningful generalization result.

An exact normalized-state and shared eight-word-sequence audit compared the
full 1,000-record candidate pool against all 231 public items and found zero
hits (`results/contamination_audit.json`). With the training file:

```bash
python scripts/contamination_audit.py \
  --train data/training_mix.jsonl \
  --jevbench-root /path/to/jevbench/datasets/public
```

## Limits

- English only.
- Up to 16 options use one pass; larger sets use the runtime's knockout readout.
- Inputs above 16,384 prompt tokens are rejected, never truncated.
- FP8 is hardware/runtime-sensitive. Rerun on the evaluator's H100 image.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
