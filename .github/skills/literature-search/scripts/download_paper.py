"""
Download a PDF from a given URL to a local directory.

Usage:
    python download_paper.py <url> <output_dir> [--filename <name.pdf>]

Example:
    python download_paper.py "https://arxiv.org/pdf/2301.12345" "research/topic/papers/" --filename "Author2023_Title.pdf"
"""

import argparse
import os
import re
import sys
from urllib.parse import urlparse, unquote

import requests


def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.strip('. ')
    return name[:200] if name else 'downloaded_paper.pdf'


def derive_filename(url: str) -> str:
    """Derive a filename from the URL if none is provided."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    basename = os.path.basename(path)
    if basename and '.' in basename:
        return sanitize_filename(basename)
    # Fallback for URLs like arxiv.org/abs/2301.12345
    parts = path.strip('/').split('/')
    if parts:
        return sanitize_filename(parts[-1]) + '.pdf'
    return 'downloaded_paper.pdf'


def download_pdf(url: str, output_dir: str, filename: str | None = None) -> str:
    """Download a PDF from url to output_dir/filename. Returns the saved path."""
    if filename is None:
        filename = derive_filename(url)
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return output_path

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print(f"Downloading: {url}")
    resp = requests.get(url, headers=headers, timeout=60, stream=True)
    resp.raise_for_status()

    content_type = resp.headers.get('Content-Type', '')
    if 'pdf' not in content_type and 'octet-stream' not in content_type:
        print(f"Warning: Content-Type is '{content_type}', may not be a PDF.")

    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved: {output_path} ({size_kb:.1f} KB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Download a PDF from a URL.')
    parser.add_argument('url', help='URL of the PDF to download')
    parser.add_argument('output_dir', help='Directory to save the PDF')
    parser.add_argument('--filename', help='Output filename (default: derived from URL)')
    args = parser.parse_args()

    try:
        path = download_pdf(args.url, args.output_dir, args.filename)
        print(f"Success: {path}")
    except requests.RequestException as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
