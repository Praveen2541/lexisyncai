
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.ingest import parse_pdf
from src.retrieve import Index
from src.generate import answer, llm
from src.abstain import confidence

QFILE = os.path.join(os.path.dirname(__file__), "questions.jsonl")
CONFIGS = {
    "bm25": ("BM25 only", "BM25"),
    "dense": ("Dense only", "Dense"),
    "hybrid": ("Hybrid (BM25 + dense, RRF)", "BM25 + dense"),
    "hybrid_rerank": ("Hybrid + cross-encoder rerank", "BM25 + dense + cross-encoder"),
}
THRESHOLDS = [round(i / 100, 2) for i in range(5, 51, 5)]


def load_questions():
    with open(QFILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def is_llm_error(result):
    return str(result.get("out", {}).get("explanation", "")).startswith("LLM error:")


def evaluate_retrieval(index, questions):
    """Measure question-level evidence coverage and latency for each retrieval mode."""
    answerable = [x for x in questions if x["answerable"]]
    covered_by_corpus = sum(covers(index.c, x) for x in answerable)
    rows = []
    for mode, (label, models) in CONFIGS.items():
        hits = {k: 0 for k in (1, 3, 5)}
        elapsed = []
        for x in answerable:
            started = time.perf_counter()
            results = index.search(x["q"], mode, k=5)
            elapsed.append((time.perf_counter() - started) * 1000)
            retrieved = [index.c[i] for i, _ in results]
            for k in hits:
                if covers(retrieved[:k], x):
                    hits[k] += 1
        rows.append({
            "Configuration": label,
            "Models loaded": models,
            "Recall@1": hits[1] / max(len(answerable), 1),
            "Recall@3": hits[3] / max(len(answerable), 1),
            "Recall@5": hits[5] / max(len(answerable), 1),
            "Latency (ms)": float(np.mean(elapsed)) if elapsed else 0.0,
        })
    return {
        "questions_used": len(questions),
        "answerable_questions": len(answerable),
        "gold_covered_questions": covered_by_corpus,
        "gold_coverage": covered_by_corpus / max(len(answerable), 1),
        "rows": rows,
    }


def evaluate_thresholds(questions, results):
    """Evaluate abstention accuracy and false-abstain rate for confidence thresholds."""
    valid_pairs = [(q, r) for q, r in zip(questions, results) if not is_llm_error(r)]
    unanswerable_count = sum(not q["answerable"] for q, _ in valid_pairs)
    answerable_count = sum(q["answerable"] for q, _ in valid_pairs)
    rows = []
    for threshold in THRESHOLDS:
        correct_unanswerable = 0
        false_abstentions = 0
        for question, result in valid_pairs:
            is_abstention = confidence(
                result["out"].get("answerable"),
                result["top"],
                result["out"].get("claims", []),
                result["ctx"],
                threshold,
                float(os.getenv("TAU_HIGH", 3)),
            ) == "INSUFFICIENT EVIDENCE"
            if not question["answerable"] and is_abstention:
                correct_unanswerable += 1
            elif question["answerable"] and is_abstention:
                false_abstentions += 1
        rows.append({
            "threshold": threshold,
            "abstention_accuracy": correct_unanswerable / max(unanswerable_count, 1),
            "false_abstain_rate": false_abstentions / max(answerable_count, 1),
        })
    return rows


def main(pdfs):
    random.seed(0); np.random.seed(0)
    chunks = []
    for f in pdfs:
        chunks += parse_pdf(f, os.path.basename(f))[0]
    idx = Index(chunks)
    Q = load_questions()
    ans = [x for x in Q if x["answerable"]]
    lines = []
    P = lambda s="": (print(s), lines.append(s))

    P("## Recall@k per retrieval configuration (answerable questions)")
    retrieval = evaluate_retrieval(idx, Q)
    P(f"- Gold evidence present in corpus: {retrieval['gold_covered_questions']}/"
      f"{retrieval['answerable_questions']} = {retrieval['gold_coverage']:.2f}")
    P("| Configuration | Models loaded | Recall@1 | Recall@3 | Recall@5 | Latency (ms) |\n"
      "|---|---|---:|---:|---:|---:|")
    for row in retrieval["rows"]:
        P(f"| {row['Configuration']} | {row['Models loaded']} | {row['Recall@1']:.3f} | "
          f"{row['Recall@3']:.3f} | {row['Recall@5']:.3f} | {row['Latency (ms)']:.1f} |")

    P("\n## Full pipeline")
    lat, cp, ab_ok, false_ab, res, stage, raw_results = [], [], 0, 0, [], {}, []
    for x in Q:
        t = time.time(); r = answer(x["q"], idx); lat.append(time.time() - t)
        raw_results.append(r)
        for k, v in r["timings"].items(): stage.setdefault(k, []).append(v)
        abst = r["confidence"] == "INSUFFICIENT EVIDENCE"
        if not x["answerable"]:
            ab_ok += abst and not is_llm_error(r)
        else:
            false_ab += abst and not is_llm_error(r)
            if not is_llm_error(r):
                cited = [r["ctx"][c] for cl in r["out"].get("claims", []) for c in cl["cites"] if c in r["ctx"]]
                if cited: cp.append(np.mean([relevant(c, x) for c in cited]))
        res.append(dict(q=x["q"], type=x["type"], answerable=x["answerable"], confidence=r["confidence"],
                        top=r["top"], tokens=r["tokens"], claims=r["out"].get("claims"),
                        conflicts=r["out"].get("conflicts"), manual_grade="TODO (1 / 0.5 / 0)"))
    valid_unanswerable = sum(
        not x["answerable"] and not is_llm_error(r) for x, r in zip(Q, raw_results)
    )
    valid_answerable = sum(
        x["answerable"] and not is_llm_error(r) for x, r in zip(Q, raw_results)
    )
    P(f"- Abstention accuracy (unanswerable): {ab_ok}/{valid_unanswerable} = "
      f"{ab_ok/max(valid_unanswerable,1):.2f}")
    P(f"- False-abstention rate (answerable): {false_ab}/{valid_answerable} = "
      f"{false_ab/max(valid_answerable,1):.2f}")
    P(f"- Citation precision (mean): {np.mean(cp) if cp else 0:.2f}")
    P(f"- Latency per question: avg {np.mean(lat):.1f}s, p95 {np.percentile(lat,95):.1f}s")
    P("- Avg per stage: " + ", ".join(f"{k} {np.mean(v):.2f}s" for k, v in stage.items()))
    P(f"- Avg tokens/question: {np.mean([r['tokens'] for r in res]):.0f}")
    threshold_rows = evaluate_thresholds(Q, raw_results)
    P("\n## Abstention-gate calibration (threshold sweep)")
    P("| Threshold | Abstention accuracy | False-abstain rate |\n|---:|---:|---:|")
    for row in threshold_rows:
        P(f"| {row['threshold']:.2f} | {row['abstention_accuracy']:.3f} | "
          f"{row['false_abstain_rate']:.3f} |")

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
    json.dump({
        "retrieval": retrieval,
        "threshold_sweep": threshold_rows,
        "pipeline": res,
    }, open("docs/eval_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open("docs/results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved docs/eval_results.json and docs/results.md")


if __name__ == "__main__":
    main(sys.argv[1:])
