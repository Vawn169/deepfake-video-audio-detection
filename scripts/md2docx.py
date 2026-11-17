"""Chuyen docs/BAO_CAO_v2_draft.md sang Word (tieu de, doan, danh sach, bang, **dam**, `ma`). Khong can pandoc.
  .venv/bin/python scripts/md2docx.py docs/BAO_CAO_v2_draft.md reports/BAO_CAO_tong_hop_GVHD_v2.docx"""
import re, sys
from docx import Document
from docx.shared import Pt
from docx.enum.table import WD_TABLE_ALIGNMENT

def runs(par, text):
    for part in re.split(r'(\*\*[^*]+\*\*|`[^`]+`)', text):
        if not part:
            continue
        if part.startswith('**'):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith('`'):
            r = par.add_run(part[1:-1]); r.font.name = 'Consolas'; r.font.size = Pt(9)
        else:
            par.add_run(part.replace('\\|', '|'))

def cells(line):
    return [c.strip() for c in re.split(r'(?<!\\)\|', line.strip().strip('|'))]

src, dst = sys.argv[1], sys.argv[2]
doc = Document(); st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(12)
lines = open(src, encoding='utf-8').read().splitlines(); i = 0
while i < len(lines):
    l = lines[i]
    if l.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s\-|:]+\|$', lines[i + 1].strip()):
        head = cells(l); body = []; i += 2
        while i < len(lines) and lines[i].startswith('|'):
            body.append(cells(lines[i])); i += 1
        t = doc.add_table(rows=1 + len(body), cols=len(head)); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for r, row in enumerate([head] + body):
            for c in range(len(head)):
                p = t.cell(r, c).paragraphs[0]; runs(p, row[c] if c < len(row) else '')
                for rn in p.runs:
                    rn.font.size = Pt(10); rn.bold = rn.bold or r == 0
        doc.add_paragraph(); continue
    m = re.match(r'^(#{1,4})\s+(.*)', l)
    if m:
        doc.add_heading(m.group(2).replace('**', ''), level=len(m.group(1)) - (1 if len(m.group(1)) > 1 else 0))
    elif re.match(r'^\s*[-*]\s+', l):
        lvl = (len(l) - len(l.lstrip())) // 2
        runs(doc.add_paragraph(style='List Bullet 2' if lvl else 'List Bullet'), re.sub(r'^\s*[-*]\s+', '', l))
    elif re.match(r'^\s*\d+\.\s+', l):
        runs(doc.add_paragraph(style='List Number'), re.sub(r'^\s*\d+\.\s+', '', l))
    elif l.strip():
        runs(doc.add_paragraph(), l)
    i += 1
doc.save(dst); print('Da luu', dst)
