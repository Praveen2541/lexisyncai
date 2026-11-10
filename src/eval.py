
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
random.seed(0); np.random.seed(0)
from src.ingest import parse_pdf
from src.retrieve import Index
from src.generate import answer, llm

QFILE = os.path.join(os.path.dirname(__file__), "questions.jsonl")


def covers(chunks, x):
    """Retrieval hit: all gold keywords appear in the retrieved text (or a gold page is retrieved)."""
    blob = " ".join(c["text"].lower() for c in chunks)
    kw = [k.lower() for k in x.get("gold_keywords", [])]
    if kw and all(k in blob for k in kw):
        return True
    gp = set(x.get("gold_pages", []))
    return bool(gp) and any(set(range(c["page"], c["page_end"] + 1)) & gp for c in chunks)


def relevant(c, x):
    kw = [k.lower() for k in x.get("gold_keywords", [])]
    if any(k in c["text"].lower() for k in kw):
        return True
    return bool(set(range(c["page"], c["page_end"] + 1)) & set(x.get("gold_pages", [])))


def main(pdfs):
    chunks = []
    for f in pdfs:
        chunks += parse_pdf(f, os.path.basename(f))[0]
    idx = Index(chunks)
    Q = [json.loads(l) for l in open(QFILE, encoding="utf-8") if l.strip()]
    ans = [x for x in Q if x["answerable"]]
    lines = []
    P = lambda s="": (print(s), lines.append(s))

    P("## Recall@k per retrieval configuration (answerable questions)")
    P("| config | R@1 | R@3 | R@5 |\n|---|---|---|---|")
    for mode in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        row = []
        for k in (1, 3, 5):
            hit = [covers([idx.c[i] for i, _ in idx.search(x["q"], mode, k=k)], x) for x in ans]
            row.append(f"{np.mean(hit):.2f}")
        P(f"| {mode} | " + " | ".join(row) + " |")

    P("\n## Full pipeline")
    lat, cp, ab_ok, false_ab, res, stage = [], [], 0, 0, [], {}
    for x in Q:
        t = time.time(); r = answer(x["q"], idx); lat.append(time.time() - t)
        for k, v in r["timings"].items(): stage.setdefault(k, []).append(v)
        abst = r["confidence"] == "INSUFFICIENT EVIDENCE"
        if not x["answerable"]:
            ab_ok += abst
        else:
            false_ab += abst
            cited = [r["ctx"][c] for cl in r["out"].get("claims", []) for c in cl["cites"] if c in r["ctx"]]
            if cited: cp.append(np.mean([relevant(c, x) for c in cited]))
        res.append(dict(q=x["q"], type=x["type"], answerable=x["answerable"], confidence=r["confidence"],
                        top=r["top"], tokens=r["tokens"], claims=r["out"].get("claims"),
                        conflicts=r["out"].get("conflicts"), manual_grade="TODO (1 / 0.5 / 0)"))
    nu = sum(not x["answerable"] for x in Q)
    P(f"- Abstention accuracy (unanswerable): {ab_ok}/{nu} = {ab_ok/max(nu,1):.2f}")
    P(f"- False-abstention rate (answerable): {false_ab}/{len(ans)} = {false_ab/max(len(ans),1):.2f}")
    P(f"- Citation precision (mean): {np.mean(cp) if cp else 0:.2f}")
    P(f"- Latency per question: avg {np.mean(lat):.1f}s, p95 {np.percentile(lat,95):.1f}s")
    P("- Avg per stage: " + ", ".join(f"{k} {np.mean(v):.2f}s" for k, v in stage.items()))
    P(f"- Avg tokens/question: {np.mean([r['tokens'] for r in res]):.0f}")

    P("\n## No-retrieval baseline (LLM alone) vs pipeline - grade both manually")
    for x, r in zip(ans, [r for r in res if r["answerable"]]):
        try:
            o, _ = llm('Answer the legal question briefly. JSON {"answer": str}', x["q"])
            r["no_retrieval_answer"] = o.get("answer")
        except Exception as e:
            r["no_retrieval_answer"] = f"error: {e}"
        r["no_retrieval_grade"] = "TODO (1 / 0.5 / 0)"
        P(f"- {x['q']}\n    NO-RETRIEVAL: {r['no_retrieval_answer']}")

    os.makedirs("docs", exist_ok=True)
    json.dump(res, open("docs/eval_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open("docs/results.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\nSaved docs/eval_results.json and docs/results.md")


if __name__ == "__main__":
    main(sys.argv[1:])
