"""Console entrypoint: launches the FastAPI/uvicorn server.

Usage:  python ui/launch_ui.py  [--port 8765] [--host 127.0.0.1]

When this process exits (Ctrl+C, window closed, parent shell crashed),
the Job Object held by ProcessManager forcibly kills every child
process (evolution main + 16 workers + watcher) — no orphans possible.
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BPS = _HERE.parent
if str(_BPS) not in sys.path:
    sys.path.insert(0, str(_BPS))

import uvicorn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"[ui] starting {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    # delay import so we can put _BPS on path first
    from ui.ui_server import app
    uvicorn.run(app, host=args.host, port=args.port,
                log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
