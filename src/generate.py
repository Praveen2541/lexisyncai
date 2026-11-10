
import os, json, time
from dotenv import load_dotenv
from src.abstain import confidence

load_dotenv()
TAU_LOW = float(os.getenv("TAU_LOW", -2))
TAU_HIGH = float(os.getenv("TAU_HIGH", 3))
_cli = None


def _client():
    global _cli
    if _cli is None:
        from openai import OpenAI
        _cli = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
    return _cli


def llm(system, user):
    r = _client().chat.completions.create(
        model=os.getenv("LLM_MODEL"), temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content), (r.usage.total_tokens if r.usage else 0)


def decompose(q):
    try:
        o, t = llm('Split the legal question into 1-3 standalone search queries (one per fact needed). '
                   'Return JSON {"queries":["..."]}.', q)
        return (o.get("queries") or [q])[:3], t
    except Exception:
        return [q], 0


SYS = """You are a legal analyst. Answer ONLY from the numbered context. Never use outside knowledge.
Return JSON: {"answerable": bool, "claims":[{"text": str, "cites": ["C1"]}],
"conflicts": str, "explanation": str}.
Rules:
- Every claim must cite >=1 context ID (e.g. "C2"). Mention page/clause in the claim text.
- If the context does not contain the evidence, set answerable=false and claims=[].
- If sources disagree (different documents/dates/values), state BOTH values with document and page in "conflicts",
  and say which one you used and why (e.g. most recent / more specific). Else conflicts="".
- "explanation": one or two sentences on why this evidence supports the answer."""


def answer(q, index, multihop=True):
    T, toks = {}, 0
    t = time.time()
    qs, tk = decompose(q) if multihop else ([q], 0)
    toks += tk
    T["decompose"] = time.time() - t

    t = time.time()
    seen, hits = set(), []
    for sq in qs:
        for i, sc in index.search(sq):
            if i not in seen:
                seen.add(i)
                hits.append((i, sc))
    hits = sorted(hits, key=lambda x: -x[1])[:8]
    T["retrieve+rerank"] = time.time() - t

    ctx = {f"C{n+1}": index.c[i] for n, (i, _) in enumerate(hits)}
    block = "\n".join(
        f"[{k}] (Doc: {c['doc']}, p.{c['page']}, {c['clause']}) {c['text']}" for k, c in ctx.items())
    t = time.time()
    try:
        out, tk = llm(SYS, f"Context:\n{block}\n\nQuestion: {q}")
    except Exception as e:
        out, tk = {"answerable": False, "claims": [], "explanation": f"LLM error: {e}"}, 0
    toks += tk
    T["generate"] = time.time() - t

    top = hits[0][1] if hits else -99
    conf = confidence(out.get("answerable"), top, out.get("claims", []), ctx, TAU_LOW, TAU_HIGH)
    return dict(confidence=conf, out=out, ctx=ctx, top=top, tokens=toks, timings=T, subqueries=qs,
                searched=sorted({c["doc"] for c in index.c}), n_chunks=len(index.c))
