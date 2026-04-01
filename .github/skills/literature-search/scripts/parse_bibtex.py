"""
Extract BibTeX entries from extracted paper text.
Attempts to parse author, title, year, venue information from text content
and produce valid BibTeX entries.

Usage:
    python parse_bibtex.py <extracted_text_file> [--output <output.bib>]

Example:
    python parse_bibtex.py "research/topic/papers/extracted_Author2023.txt" --output "research/topic/references.bib"
"""

import argparse
import os
import re
import sys


def extract_metadata(text: str) -> dict:
    """Attempt to extract paper metadata from the first portion of extracted text."""
    lines = text.strip().split('\n')
    metadata = {
        'title': '',
        'authors': '',
        'year': '',
        'venue': '',
        'doi': '',
        'abstract': '',
    }

    # Try to find title (usually the first substantial line)
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 20 and not line.startswith(('Abstract', 'IEEE', 'http', '©')):
            metadata['title'] = line
            break

    # Try to find year
    year_match = re.search(r'\b(19|20)\d{2}\b', text[:2000])
    if year_match:
        metadata['year'] = year_match.group()

    # Try to find DOI
    doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', text[:3000])
    if doi_match:
        metadata['doi'] = doi_match.group().rstrip('.')

    # Try to find authors (lines before title with comma/and patterns)
    author_pattern = re.compile(
        r'(?:[A-Z][a-z]+ [A-Z][a-z]+(?:,\s*(?:and\s+)?)?)+', re.MULTILINE
    )
    author_matches = author_pattern.findall(text[:1500])
    if author_matches:
        # Take the longest match as likely author list
        metadata['authors'] = max(author_matches, key=len)

    # Try to find venue
    venue_patterns = [
        r'IEEE\s+Transactions\s+on\s+[\w\s]+',
        r'IEEE\s+Journal\s+on\s+[\w\s]+',
        r'IEEE\s+(?:ICC|Globecom|ISIT|VTC|WCNC|PIMRC)\s*\d*',
        r'(?:Proceedings|Proc\.)\s+(?:of\s+)?(?:the\s+)?\d+\w*\s+[\w\s]+',
    ]
    for pattern in venue_patterns:
        venue_match = re.search(pattern, text[:3000])
        if venue_match:
            metadata['venue'] = venue_match.group().strip()
            break

    # Extract abstract
    abs_match = re.search(
        r'Abstract[—:\s]+(.+?)(?=\n\s*(?:Index Terms|Keywords|I\.\s|1\.\s|Introduction))',
        text[:5000],
        re.DOTALL | re.IGNORECASE,
    )
    if abs_match:
        metadata['abstract'] = ' '.join(abs_match.group(1).split())

    return metadata


def generate_cite_key(metadata: dict) -> str:
    """Generate a citation key from metadata."""
    author_part = ''
    if metadata['authors']:
        first_author = metadata['authors'].split(',')[0].split(' and ')[0].strip()
        parts = first_author.split()
        if parts:
            author_part = parts[-1]  # Last name
    year_part = metadata['year'] if metadata['year'] else 'XXXX'

    title_word = ''
    if metadata['title']:
        # Take first meaningful word from title
        skip_words = {'a', 'an', 'the', 'on', 'for', 'of', 'in', 'with', 'and', 'to'}
        for word in metadata['title'].split():
            clean = re.sub(r'[^a-zA-Z]', '', word)
            if clean.lower() not in skip_words and len(clean) > 2:
                title_word = clean
                break

    return f"{author_part}{year_part}{title_word}" if author_part else f"paper{year_part}{title_word}"


def format_bibtex(metadata: dict, cite_key: str) -> str:
    """Format metadata as a BibTeX entry."""
    entry_type = 'article'
    if any(kw in metadata.get('venue', '').lower() for kw in ['proceedings', 'proc', 'conference', 'icc', 'globecom', 'isit']):
        entry_type = 'inproceedings'

    lines = [f"@{entry_type}{{{cite_key},"]
    if metadata['title']:
        lines.append(f'  title = {{{metadata["title"]}}},')
    if metadata['authors']:
        lines.append(f'  author = {{{metadata["authors"]}}},')
    if metadata['year']:
        lines.append(f'  year = {{{metadata["year"]}}},')
    if metadata['venue']:
        field = 'booktitle' if entry_type == 'inproceedings' else 'journal'
        lines.append(f'  {field} = {{{metadata["venue"]}}},')
    if metadata['doi']:
        lines.append(f'  doi = {{{metadata["doi"]}}},')
    lines.append('}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Extract BibTeX from paper text.')
    parser.add_argument('input', help='Path to extracted text file')
    parser.add_argument('--output', help='Path to output .bib file (default: stdout)')
    parser.add_argument('--append', action='store_true', help='Append to output file instead of overwriting')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    metadata = extract_metadata(text)
    cite_key = generate_cite_key(metadata)
    bibtex = format_bibtex(metadata, cite_key)

    if args.output:
        mode = 'a' if args.append else 'w'
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, mode, encoding='utf-8') as f:
            if args.append:
                f.write('\n\n')
            f.write(bibtex)
        print(f"BibTeX written to: {args.output}")
    else:
        print(bibtex)

    print(f"\nExtracted metadata:")
    for key, value in metadata.items():
        if value and key != 'abstract':
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
