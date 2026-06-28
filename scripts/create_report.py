#!/usr/bin/env python3
"""Convert both English and Vietnamese reports from markdown to tex/pdf.

Run from the repository root so that relative image paths (output/*.png) resolve.
"""
import subprocess, os, sys

# Ensure we run from the repo root (parent of scripts/)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(repo_root)
os.makedirs('report', exist_ok=True)

for lang, suffix in [('Vietnamese', ''), ('English', '-en')]:
    md = f'report/bao-cao{suffix}.md'
    tex = f'report/bao-cao{suffix}.tex'
    pdf = f'report/bao-cao{suffix}.pdf'

    if not os.path.exists(md):
        print(f"  SKIP {md} (not found)")
        continue

    # TeX output
    subprocess.run([
        'pandoc', md, '-o', tex,
        '--standalone',
        '--number-sections',
        '-V', 'geometry:margin=1in',
        '-V', 'lang=en',
        '-V', 'mainfont=DejaVu Sans',
    ], check=True)
    print(f"  {tex}")

    # PDF output (xelatex supports Unicode)
    subprocess.run([
        'pandoc', md, '-o', pdf,
        '--pdf-engine=xelatex',
        '--standalone',
        '--number-sections',
        '-V', 'geometry:margin=1in',
        '-V', 'lang=en',
        '-V', 'mainfont=DejaVu Sans',
    ], check=True)
    print(f"  {pdf}")

print("Done")
