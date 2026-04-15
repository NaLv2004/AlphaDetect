"""Download automl_zero folder from google-research GitHub repo using the API."""
import os
import json
import ssl
import urllib.request
import urllib.error
import time

BASE = "https://api.github.com/repos/google-research/google-research/contents/automl_zero"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "automl_zero")

# Create SSL context that handles connection issues
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def get_contents(api_url, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"Rate limited, waiting 60s...")
                time.sleep(60)
            else:
                raise
        except Exception as e:
            print(f"  Retry {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise

def download_file(url, dest_path, retries=5):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                with open(dest_path, 'wb') as f:
                    f.write(resp.read())
            return
        except Exception as e:
            print(f"    Download retry {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise

def process_dir(api_url, local_dir):
    items = get_contents(api_url)
    if not isinstance(items, list):
        print(f"Unexpected response: {items}")
        return
    for item in items:
        name = item["name"]
        if item["type"] == "dir":
            print(f"  [DIR] {name}/")
            process_dir(item["url"], os.path.join(local_dir, name))
        elif item["type"] == "file":
            dest = os.path.join(local_dir, name)
            if os.path.exists(dest) and os.path.getsize(dest) == item.get("size", -1):
                print(f"  [SKIP] {name} (already exists)")
                continue
            print(f"  [FILE] {name} ({item.get('size', '?')} bytes)")
            download_file(item["download_url"], dest)

if __name__ == "__main__":
    print(f"Downloading automl_zero to {DEST}")
    os.makedirs(DEST, exist_ok=True)
    process_dir(BASE, DEST)
    print("Done!")
