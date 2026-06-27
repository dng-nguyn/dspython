#!/usr/bin/env python3
"""Convert the English report from markdown to tex/pdf."""
import subprocess, os
os.makedirs('report', exist_ok=True)
subprocess.run(['pandoc', 'report/bao-cao.md', '-o', 'report/bao-cao.tex',
                '--standalone', '--number-sections', '-V', 'geometry:margin=1in'], check=True)
print("report/bao-cao.tex")
subprocess.run(['pandoc', 'report/bao-cao.md', '-o', 'report/bao-cao.pdf',
                '--pdf-engine=xelatex', '--standalone', '--number-sections',
                '-V', 'geometry:margin=1in'], check=True)
print("report/bao-cao.pdf")
