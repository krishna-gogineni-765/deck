"""Display and edit a taxonomy in the browser."""

from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from deck.label import plan_label
from deck.models import MODELS
from deck.taxonomy import Class, Taxonomy, description_is_weak, load_taxonomy, save_taxonomy


def render(taxonomy: Taxonomy, *, notice: str = "", route_text: str = "") -> str:
    rows = []
    for item in taxonomy.classes:
        weak = "needs a description" if description_is_weak(item) else "ready"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(item.id)}</code><div class='meta'>{weak}</div></td>"
            "<td>"
            f"<form method='post' action='/classes/{html.escape(item.id)}'>"
            f"<textarea name='description' rows='3'>{html.escape(item.description)}</textarea>"
            f"<input name='exclusions' value='{html.escape(item.exclusions)}' "
            "placeholder='exclusions'>"
            f"<input name='example' value='{html.escape(item.example)}' "
            "placeholder='one positive example'>"
            "<button type='submit'>Save class</button>"
            "</form></td></tr>"
        )
    model_rows = "".join(
        f"<li><code>{html.escape(model.id)}</code> — {html.escape(model.role)} — "
        f"{html.escape(model.summary)}</li>"
        for model in MODELS.values()
    )
    plan = ""
    if route_text.strip():
        result = plan_label(route_text, taxonomy)
        reasons = ", ".join(result.reasons) or "none"
        plan = (
            f"<p>Label with <strong>{html.escape(result.model)}</strong> "
            f"(score {result.score}; {html.escape(reasons)}). "
            f"Decide with <strong>{html.escape(result.decision_model)}</strong>.</p>"
        )
    message = f"<p class='notice'>{html.escape(notice)}</p>" if notice else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>deck · {html.escape(taxonomy.id)}</title>
<style>
body {{ font: 16px/1.4 system-ui, sans-serif; margin: 2rem auto; max-width: 920px; color: #1c1917; }}
textarea, input {{ width: 100%; box-sizing: border-box; margin: 0.25rem 0; font: inherit; }}
table {{ width: 100%; border-collapse: collapse; }}
td {{ vertical-align: top; padding: 0.75rem 0.5rem; border-top: 1px solid #e7e5e4; }}
.meta {{ color: #78716c; font-size: 0.85rem; }}
.notice {{ background: #f5f5f4; padding: 0.5rem 0.75rem; }}
button {{ font: inherit; }}
</style></head><body>
<h1>deck</h1>
<p>Taxonomy <strong>{html.escape(taxonomy.id)}</strong>. {html.escape(taxonomy.question)}</p>
{message}
<h2>Classes</h2>
<table>{''.join(rows)}</table>
<h2>Label an example</h2>
<form method="post" action="/route">
<textarea name="state" rows="5" placeholder="Paste an example">{html.escape(route_text)}</textarea>
<button type="submit">Choose a labeler</button>
</form>
{plan}
<h2>Models</h2>
<ul>{model_rows}</ul>
</body></html>"""


def _field(form: dict[str, list[str]], name: str) -> str:
    return (form.get(name) or [""])[0].strip()


class TaxonomyApp:
    def __init__(self, path: Path):
        self.path = path
        self.taxonomy = load_taxonomy(path)

    def page(self, *, notice: str = "", route_text: str = "") -> bytes:
        return render(self.taxonomy, notice=notice, route_text=route_text).encode()

    def save_class(self, class_id: str, form: dict[str, list[str]]) -> str:
        current = self.taxonomy.get(class_id)
        self.taxonomy = self.taxonomy.replace(
            Class(
                id=current.id,
                description=_field(form, "description"),
                exclusions=_field(form, "exclusions"),
                example=_field(form, "example"),
            )
        )
        save_taxonomy(self.taxonomy, self.path)
        return f"Saved {class_id}."


def serve_taxonomy(path: Path, host: str, port: int) -> None:
    app = TaxonomyApp(path)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/":
                self._send(b"Not found", 404)
                return
            self._send(app.page())

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            form = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
            target = self.path.split("?", 1)[0]
            if target == "/route":
                self._send(app.page(route_text=_field(form, "state")))
                return
            prefix = "/classes/"
            if not target.startswith(prefix):
                self._send(b"Not found", 404)
                return
            class_id = target[len(prefix):]
            try:
                notice = app.save_class(class_id, form)
            except KeyError:
                self._send(b"Unknown class", 404)
                return
            self._send(app.page(notice=notice))

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"deck taxonomy on http://{host}:{port}/", flush=True)
    server.serve_forever()
