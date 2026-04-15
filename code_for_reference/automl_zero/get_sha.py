"""Calculate SHA256 of protobuf 3.19.6 archive."""
import urllib.request, ssl, hashlib, sys

url = "https://github.com/protocolbuffers/protobuf/archive/v3.19.6.zip"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print(f"Downloading: {url}")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
resp = urllib.request.urlopen(req, context=ctx)
data = resp.read()
sha = hashlib.sha256(data).hexdigest()
print(f"Size: {len(data)} bytes")
print(f"SHA256: {sha}")
