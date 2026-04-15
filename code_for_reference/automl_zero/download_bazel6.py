"""Download a newer Bazel that supports VS 2022."""
import urllib.request
import ssl
import os
import sys

# Bazel 6.5.0 LTS - supports VS 2022
version = "6.5.0"
url = f"https://releases.bazel.build/{version}/release/bazel-{version}-windows-x86_64.exe"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"bazel-{version}.exe")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print(f"Downloading Bazel {version} from:\n  {url}")
print(f"Saving to: {out}")

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
resp = urllib.request.urlopen(req, context=ctx)
total = int(resp.headers.get("Content-Length", 0))
print(f"Total size: {total / 1024 / 1024:.1f} MB")

downloaded = 0
chunk_size = 1024 * 256
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
