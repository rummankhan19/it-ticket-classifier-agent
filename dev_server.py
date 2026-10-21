"""Run the agent locally without the Vercel CLI: serves public/ and POST /api/chat.

Usage:  python dev_server.py   ->  http://localhost:8000
Reads GEMINI_API_KEY (and optional GEMINI_MODEL) from the environment or a .env file.
"""
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent

env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from api.chat import handler as ChatHandler  # noqa: E402  (needs env loaded first)


class DevHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "public"), **kwargs)

    def do_POST(self):
        if self.path.startswith("/api/chat"):
            return ChatHandler.do_POST(self)
        self.send_error(404)

    _send = ChatHandler._send


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"TicketSense running at http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), DevHandler).serve_forever()
