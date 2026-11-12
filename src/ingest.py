
import re
from collections import Counter
import fitz  # PyMuPDF

KW = ("article", "section", "clause")
HEAD = re.compile(
    r'^\s*(Article\s+\d+|Section\s+\d+(?:\.\d+)*|Clause\s+\d+(?:\.\d+)*|'
    r'\d+(?:\.\d+)+|\d{1,2}\.|\(\d+\)|\([a-z]\))\s')
_norm = lambda t: re.sub(r'\d+', '#', t.strip().lower())[:60]


def _order_blocks(bl, W):
    """Two-column pages: left column then right column. Otherwise top-to-bottom."""
    L = [b for b in bl if b[2] <= W / 2 + 10]
    R = [b for b in bl if b[0] >= W / 2 - 10 and b not in L]
    if len(L) >= 3 and len(R) >= 3 and len(L) + len(R) >= 0.8 * len(bl):
        return sorted(L, key=lambda b: b[1]) + sorted(R, key=lambda b: b[1])
    return sorted(bl, key=lambda b: (round(b[1]), b[0]))


def parse_pdf(path, name, max_words=200):
    """Return (chunks, page_stats). Chunk = dict(doc, page, page_end, clause, text, bbox, n)."""
    d = fitz.open(path)
    n = len(d)
    raw = [(p, [b for b in p.get_text("blocks") if b[6] == 0 and b[4].strip()]) for p in d]

    # headers/footers: text repeating in top/bottom 8% on >=40% of pages
    cnt = Counter()
    for p, bl in raw:
        H = p.rect.height
        for b in bl:
            if b[1] < .08 * H or b[3] > .92 * H:
                cnt[_norm(b[4])] += 1

    chunks, pages, art, cur = [], [], "", None
    for i, (p, bl) in enumerate(raw):
        H, W = p.rect.height, p.rect.width
        bl = [b for b in bl
              if not ((b[1] < .08 * H or b[3] > .92 * H) and cnt[_norm(b[4])] >= max(2, .4 * n))]
        bl = _order_blocks(bl, W)
        chars = sum(len(b[4]) for b in bl)
        try:
            ntab = len(p.find_tables().tables)
        except Exception:
            ntab = 0
        pages.append(dict(page=i + 1, chars=chars, tables=ntab,
                          scanned=chars < 20 and len(p.get_images()) > 0))
        for b in bl:
            text = " ".join(b[4].split())
            m = HEAD.match(text)
            if m:
                tok = m.group(1).strip()
                if tok.lower().startswith(KW):
                    art = tok
                    clause = tok
                else:
                    clause = f"{art} {tok}".strip()
                new = True
            else:
                clause = cur["clause"] if cur else ""
                new = cur is None or cur["n"] > max_words
            if new:
                cur = dict(doc=name, page=i + 1, page_end=i + 1, clause=clause, text="", bbox=[], n=0)
                chunks.append(cur)
            cur["text"] += " " + text
            cur["n"] += len(text.split())
            cur["page_end"] = i + 1
            cur["bbox"].append((i + 1, *b[:4]))
    for c in chunks:
        c["text"] = c["text"].strip()
    return [c for c in chunks if c["text"]], pages
