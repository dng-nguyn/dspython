#!/usr/bin/env python3
"""Convert both English and Vietnamese reports from markdown to tex/pdf."""
import subprocess, os
os.makedirs('report', exist_ok=True)

for lang, suffix in [('Vietnamese', ''), ('English', '-en')]:
    md = f'report/bao-cao{suffix}.md'
    tex = f'report/bao-cao{suffix}.tex'
    pdf = f'report/bao-cao{suffix}.pdf'
    
    subprocess.run(['pandoc', md, '-o', tex,
                    '--standalone', '--number-sections', '-V', 'geometry:margin=1in'], check=True)
    print(f"  {tex}")
    
    subprocess.run(['pandoc', md, '-o', pdf,
                    '--pdf-engine=xelatex', '--standalone', '--number-sections',
                    '-V', 'geometry:margin=1in'], check=True)
    print(f"  {pdf}")

print("Done")
