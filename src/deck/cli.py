"""deck command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deck.describe import apply_descriptions, description_prompt
from deck.label import plan_label
from deck.models import MODELS
from deck.taxonomy import description_is_weak, load_taxonomy, save_taxonomy


def _cmd_models(_args: argparse.Namespace) -> int:
    for model in MODELS.values():
        weights = f" weights={model.weights}" if model.weights else ""
        print(f"{model.id}\t{model.role}\t{model.runtime}\t{model.summary}{weights}")
    return 0


def _cmd_taxonomy_show(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    print(json.dumps(taxonomy.to_dict(), indent=2))
    weak = [item.id for item in taxonomy.classes if description_is_weak(item)]
    if weak:
        print("needs descriptions: " + ", ".join(weak), file=sys.stderr)
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    if args.apply:
        updated = apply_descriptions(
            taxonomy, json.loads(Path(args.apply).read_text(encoding="utf-8"))
        )
        save_taxonomy(updated, args.taxonomy)
        print(f"updated {args.taxonomy}")
        return 0
    print(description_prompt(taxonomy))
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    plan = plan_label(args.state, taxonomy)
    print(
        json.dumps(
            {
                "label_model": plan.model,
                "decision_model": plan.decision_model,
                "score": plan.score,
                "reasons": list(plan.reasons),
            },
            indent=2,
        )
    )
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from deck.app import serve_taxonomy

    serve_taxonomy(Path(args.taxonomy), args.host, args.port)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from deck.serve import main as serve_main

    forwarded = ["--host", args.host, "--port", str(args.port), "--name", args.name]
    if args.adapter:
        forwarded.extend(["--adapter", args.adapter])
    if args.adapter_revision:
        forwarded.extend(["--adapter-revision", args.adapter_revision])
    if not args.graphs:
        forwarded.append("--no-graphs")
    return serve_main(forwarded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deck", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="list models")
    models.set_defaults(func=_cmd_models)

    taxonomy = commands.add_parser("taxonomy", help="show a taxonomy")
    taxonomy.add_argument("taxonomy")
    taxonomy.set_defaults(func=_cmd_taxonomy_show)

    describe = commands.add_parser("describe", help="write or apply class descriptions")
    describe.add_argument("taxonomy")
    describe.add_argument("--apply", help="JSON file of teacher descriptions to save")
    describe.set_defaults(func=_cmd_describe)

    route = commands.add_parser("route", help="choose potion or the teacher for an example")
    route.add_argument("taxonomy")
    route.add_argument("--state", required=True)
    route.set_defaults(func=_cmd_route)

    ui = commands.add_parser("ui", help="display and edit a taxonomy")
    ui.add_argument("taxonomy")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8080)
    ui.set_defaults(func=_cmd_ui)

    serve = commands.add_parser("serve", help="serve the deck4b model")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8090)
    serve.add_argument("--name", default="deck4b")
    serve.add_argument("--adapter", default=None)
    serve.add_argument("--adapter-revision", default=None)
    serve.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=True)
    serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
