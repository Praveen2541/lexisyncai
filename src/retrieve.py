
import re
import numpy as np
from rank_bm25 import BM25Okapi

QP = "Represent this sentence for searching relevant passages: "
tok = lambda s: re.findall(r"\d+(?:\.\d+)+|\w+", s.lower())  # keeps "7.2" as ONE token
_M = {}


def _emb():
    if "emb" not in _M:
        from sentence_transformers import SentenceTransformer
        _M["emb"] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _M["emb"]


def _ce():
    if "ce" not in _M:
        from sentence_transformers import CrossEncoder
        _M["ce"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _M["ce"]


class Index:
    def __init__(s, chunks):
        s.c = chunks
        s.txt = [f"{c['doc']} | {c['clause']}: {c['text']}" for c in chunks]
        s.bm = BM25Okapi([tok(t) for t in s.txt])
        s.emb = _emb().encode(s.txt, normalize_embeddings=True, batch_size=32)

    def _bm(s, q, k):
        return list(np.argsort(-s.bm.get_scores(tok(q)))[:k])

    def _dn(s, q, k):
        v = _emb().encode(QP + q, normalize_embeddings=True)
        return list(np.argsort(-(s.emb @ v))[:k])

    def search(s, q, mode="hybrid_rerank", k=6, pool=30):
        """mode: bm25 | dense | hybrid | hybrid_rerank. Returns [(chunk_idx, score)]."""
        if mode == "bm25":
            return [(int(i), 0.0) for i in s._bm(q, k)]
        if mode == "dense":
            return [(int(i), 0.0) for i in s._dn(q, k)]
        rrf = {}
        for lst in (s._bm(q, pool), s._dn(q, pool)):
            for r, i in enumerate(lst):
                rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + r)
        refs = re.findall(r"\d+(?:\.\d+)+|article\s+\d+", q.lower())  # clause-reference boost
        if refs:
            for i, c in enumerate(s.c):
                if any(r in c["clause"].lower() for r in refs):
                    rrf[i] = rrf.get(i, 0) + .05
        cand = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])[:pool]]
        if mode == "hybrid":
            return [(i, rrf[i]) for i in cand[:k]]
        sc = _ce().predict([(q, s.txt[i]) for i in cand])
        return sorted(zip(cand, map(float, sc)), key=lambda x: -x[1])[:k]
