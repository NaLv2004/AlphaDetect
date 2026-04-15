"""Download Bazel 3.7.2 for Windows directly (faster than bazelisk's built-in downloader)."""
import urllib.request
import ssl
import os
import sys

url = "https://releases.bazel.build/3.7.2/release/bazel-3.7.2-windows-x86_64.exe"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bazel-3.7.2.exe")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print(f"Downloading Bazel 3.7.2 from:\n  {url}")
print(f"Saving to: {out}")

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
resp = urllib.request.urlopen(req, context=ctx)
total = int(resp.headers.get("Content-Length", 0))
print(f"Total size: {total / 1024 / 1024:.1f} MB")

downloaded = 0
chunk_size = 1024 * 256  # 256 KB chunks
with open(out, "wb") as f:
    while True:
        chunk = resp.read(chunk_size)
        if not chunk:
            break
        f.write(chunk)
        downloaded += len(chunk)
        pct = downloaded * 100 / total if total else 0
        sys.stdout.write(f"\r  {downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB  ({pct:.0f}%)")
        sys.stdout.flush()

print(f"\nDone! File size: {os.path.getsize(out)} bytes")
