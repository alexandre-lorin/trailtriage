import fcntl
import io
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse

from PIL import Image, ImageOps

from .database import Database
from .utils import safe_target

LABELS = {
    "interesting",
    "boring",
    "known_person",
    "unknown_person",
    "animal",
    "vehicle",
    "false_positive",
    "uncertain",
}


def load_rows(output: Path) -> list[dict]:
    report = output / "reports/results.jsonl"
    if not report.is_file():
        raise ValueError("Run a complete scan before building the gallery")
    rows = [json.loads(line) for line in report.read_text().splitlines()]
    return [r for r in rows if r.get("representative") and r.get("primary_category") != "empty"]


def build(output: Path) -> None:
    output = output.resolve()
    rows = load_rows(output)
    gallery = safe_target(output, output / "gallery")
    gallery.mkdir(exist_ok=True)
    thumbnails = safe_target(output, gallery / "thumbnails")
    thumbnails.mkdir(exist_ok=True)
    db = Database(output / "index.sqlite")
    try:
        data = []
        for index, row in enumerate(rows):
            thumb = safe_target(output, thumbnails / f"{index}.jpg")
            if (
                not thumb.exists()
                or thumb.stat().st_mtime_ns < Path(row["original_path"]).stat().st_mtime_ns
            ):
                with Image.open(row["original_path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image.thumbnail((360, 240))
                    image.save(thumb, quality=80)  # Re-encoding omits EXIF/GPS.
            cached = db.get(Path(row["original_path"]))
            data.append(
                {
                    "id": index,
                    "filename": row["filename"],
                    "timestamp": row["timestamp"],
                    "category": row["primary_category"],
                    "score": row["interest_score"],
                    "confidence": row["max_confidence"],
                    "classes": row["detected_classes"],
                    "href": "../" + quote(row["output_path"]),
                    "label": cached["label"] if cached else None,
                }
            )
        template = Path(__file__).with_name("gallery.html").read_text()
        encoded = json.dumps(data).replace("<", "\\u003c")
        safe_target(output, gallery / "index.html").write_text(
            template.replace("__DATA__", encoded)
        )
    finally:
        db.close()


def make_handler(output: Path):
    rows = load_rows(output)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def allowed_host(self) -> bool:
            return self.headers.get("Host", "").split(":")[0] in {"127.0.0.1", "localhost"}

        def respond(self, data: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if not self.allowed_host():
                self.send_error(403)
                return
            path = urlparse(self.path).path
            try:
                if path in {"/", "/index.html"}:
                    data = (output / "gallery/index.html").read_text().replace("__TOKEN__", token)
                    self.respond(data.encode(), "text/html; charset=utf-8")
                elif path.startswith("/image/"):
                    index = int(path.removeprefix("/image/"))
                    if index < 0:
                        raise IndexError
                    with Image.open(rows[index]["original_path"]) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        buffer = io.BytesIO()
                        image.save(buffer, "JPEG", quality=95)
                        self.respond(buffer.getvalue(), "image/jpeg")
                elif path.startswith("/thumbnails/"):
                    index = int(path.removeprefix("/thumbnails/").removesuffix(".jpg"))
                    if not 0 <= index < len(rows):
                        raise IndexError
                    self.respond(
                        (output / f"gallery/thumbnails/{index}.jpg").read_bytes(), "image/jpeg"
                    )
                else:
                    self.send_error(404)
            except (ValueError, IndexError, OSError):
                self.send_error(404)

        def do_POST(self) -> None:
            if (
                not self.allowed_host()
                or self.path != "/label"
                or self.headers.get("X-TrailTriage-Token") != token
            ):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError
                data = json.loads(self.rfile.read(size))
                index, label = data["id"], data["label"]
                if not isinstance(index, int) or not 0 <= index < len(rows) or label not in LABELS:
                    raise ValueError
                db = Database(output / "index.sqlite")
                try:
                    db.label(rows[index]["original_path"], label)
                finally:
                    db.close()
                self.respond(b'{"ok":true}', "application/json")
            except (ValueError, KeyError, TypeError):
                self.send_error(400)

    return Handler


def _serve(output: Path, port: int = 8765) -> None:
    output = output.resolve()
    build(output)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(output))
    print(f"Review: http://127.0.0.1:{port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def serve(output: Path, port: int = 8765) -> None:
    output = output.resolve()
    if not (output / ".trailtriage").is_file():
        raise ValueError("Not a TrailTriage output directory")
    with safe_target(output, output / ".scan.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A scan is running; wait before opening review") from exc
        _serve(output, port)
