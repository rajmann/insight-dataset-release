"""Local server for the Step 2 gold-standard labelling tool.

Serves index.html and the BLINDED sample (data/sample.json), and persists your
annotations to data/annotations.json on every change. The hidden gold key is
NEVER served, so there is no way to peek at the judge's scores from the UI.

Usage:
  python server.py [--port 8765]
then open http://localhost:8765 in a browser.
"""
from __future__ import annotations
import argparse
import json
import os
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SAMPLE = DATA / "sample.json"
ANNOT = DATA / "annotations.json"
BACKUPS = DATA / "backups"
INDEX = HERE / "index.html"


def atomic_write(path: Path, text: str):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def backup_existing(reason: str = ""):
    """Preserve the CURRENT annotations.json before it is overwritten, so no
    save (or mistake) can ever destroy prior work. Keeps the most recent 100."""
    if not ANNOT.exists() or ANNOT.stat().st_size <= 2:   # skip empty/{}
        return
    BACKUPS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = BACKUPS / f"annotations_{stamp}_{int(time.time()*1000)%1000:03d}{('_'+reason) if reason else ''}.json"
    try:
        dest.write_bytes(ANNOT.read_bytes())
    except OSError:
        return
    backups = sorted(BACKUPS.glob("annotations_*.json"))
    for old in backups[:-100]:
        old.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/sample.json":
            self._send(200, SAMPLE.read_bytes())
        elif self.path == "/annotations.json":
            body = ANNOT.read_bytes() if ANNOT.exists() else b"{}"
            self._send(200, body)
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/save":
            self._send(404, b'{"error":"not found"}')
            return
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode("utf-8")
        try:
            data = json.loads(raw)          # validate it parses
        except json.JSONDecodeError:
            self._send(400, b'{"error":"bad json"}')
            return
        backup_existing()                   # preserve prior state before overwrite
        atomic_write(ANNOT, json.dumps(data, ensure_ascii=False, indent=2))
        self._send(200, b'{"ok":true}')

    def log_message(self, *a):              # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not SAMPLE.exists():
        raise SystemExit(f"missing {SAMPLE} - run _build_sample.py first")
    DATA.mkdir(parents=True, exist_ok=True)
    backup_existing("startup")              # snapshot whatever exists before this session
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Labelling tool at http://localhost:{args.port}")
    print(f"Annotations persist to {ANNOT}  (rolling backups in {BACKUPS})")
    print("Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
