"""Test if we can reach GitHub for Bazel dependencies, and check proxy settings."""
import urllib.request
import ssl
import os
import socket

# Check proxy settings
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY"]:
    val = os.environ.get(var, "NOT SET")
    print(f"  {var} = {val}")

# Check if java is available
print("\nJava check:")
os.system("java -version 2>&1")

# Test connection to GitHub
url = "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print(f"\nTesting URL: {url}")
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, context=ctx, timeout=30)
    data = resp.read()
    print(f"SUCCESS: Downloaded {len(data)} bytes")
except Exception as e:
    print(f"FAILED: {e}")
    
# Test raw socket connection
print("\nTesting raw socket to github.com:443...")
try:
    sock = socket.create_connection(("github.com", 443), timeout=10)
    print("  Socket connected OK")
    sock.close()
except Exception as e:
    print(f"  Socket FAILED: {e}")
