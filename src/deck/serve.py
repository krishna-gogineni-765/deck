"""Serve the deck4b model as a TypeSafe-compatible FP8 endpoint."""

from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path

import numpy as np

import torch

BASE_MODEL = "alibiserikbay/JevK5"
BASE_REVISION = "c4f7fdb3aeab5582336406e78d3bef11bf98833d"
DEFAULT_ADAPTER = "krishna765/deck-4b-v1.0"
DEFAULT_ADAPTER_REVISION = "81e9f5e2cc701f86b1be700dfb49f6549ad83f39"
DEFAULT_NAME = "deck4b"


def _adapter_dir(adapter: str | Path, revision: str | None) -> Path:
    """A local adapter folder, or a pinned snapshot of a Hub adapter repo."""
    path = Path(adapter)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(adapter), revision=revision))


def load_fp8_model(
    adapter: str | Path = DEFAULT_ADAPTER,
    *,
    adapter_revision: str | None = DEFAULT_ADAPTER_REVISION,
    graphs: bool = True,
):
    """Load the pinned bf16 base, merge the LoRA, then quantize transformer weights."""
    from huggingface_hub import snapshot_download
    from jevk5 import JevK5
    from peft import PeftModel
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    adapter_dir = _adapter_dir(adapter, adapter_revision)
    temperature = float(
        json.loads((adapter_dir / "joint_config.json").read_text())["temperature"]
    )
    source = snapshot_download(BASE_MODEL, revision=BASE_REVISION)
    model = JevK5(source, graphs=False, temperature=temperature)
    model.model = (
        PeftModel.from_pretrained(model.model, str(adapter_dir))
        .merge_and_unload()
        .eval()
    )
    quantize_(model.model, Float8WeightOnlyConfig())
    slot_indices = torch.tensor(model.slots, device=model.device)

    def _final_hidden(self, ids: torch.Tensor, last: torch.Tensor):
        hidden = self.model.model(
            input_ids=ids, use_cache=False
        ).last_hidden_state
        return hidden[
            torch.arange(ids.shape[0], device=ids.device), last
        ]

    def _fp8_slot_logits(self, ids: torch.Tensor, last: torch.Tensor):
        final_hidden = _final_hidden(self, ids, last)
        # Apply the quantized lm_head before selecting A-P. Dequantizing just
        # those rows is faster but changed two public decisions in validation.
        return self.model.lm_head(final_hidden).index_select(1, slot_indices)

    @torch.inference_mode()
    def _capture(self, lengths=None):
        from jevk5.runtime import GRAPH_LENGTHS

        for length in lengths or GRAPH_LENGTHS:
            ids = torch.zeros(
                (1, length), dtype=torch.long, device=self.device
            )
            last = torch.full(
                (1,), length - 1, dtype=torch.long, device=self.device
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _final_hidden(self, ids, last)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                hidden = _final_hidden(self, ids, last)
            self.graphs[length] = (graph, ids, last, hidden)

    @torch.inference_mode()
    def _letter_logits(self, ids: list[int], count: int) -> np.ndarray:
        if len(ids) > 16_384:
            raise ValueError(
                f"prompt has {len(ids)} tokens; maximum is 16384"
            )
        fits = [length for length in self.graphs if length >= len(ids)]
        if fits:
            graph, static_ids, last, hidden = self.graphs[min(fits)]
            static_ids.zero_()
            static_ids[0, : len(ids)] = torch.tensor(
                ids, device=self.device
            )
            last.fill_(len(ids) - 1)
            graph.replay()
        else:
            tensor = torch.tensor([ids], device=self.device)
            last = torch.tensor([len(ids) - 1], device=self.device)
            hidden = _final_hidden(self, tensor, last)
        logits = self.model.lm_head(hidden).index_select(1, slot_indices)
        return logits[0, :count].float().cpu().numpy()

    model._slot_logits = types.MethodType(_fp8_slot_logits, model)
    model.capture = types.MethodType(_capture, model)
    model.letter_logits = types.MethodType(_letter_logits, model)
    if graphs and os.environ.get("JEVK5_GRAPHS", "1") != "0":
        model.capture()
    return model


def main(argv: list[str] | None = None) -> int:
    from http.server import ThreadingHTTPServer

    from jevk5.server import make_handler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="local adapter folder or Hugging Face adapter repo id",
    )
    parser.add_argument("--adapter-revision", default=DEFAULT_ADAPTER_REVISION)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument(
        "--graphs", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("the FP8 submission runtime requires an NVIDIA GPU")
    model = load_fp8_model(
        args.adapter, adapter_revision=args.adapter_revision, graphs=args.graphs
    )
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(model, args.name)
    )
    print(
        f"serving {args.name} on "
        f"http://{args.host}:{args.port}/v1/systemone",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
