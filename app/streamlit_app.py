import sys, os, tempfile, time, base64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, fitz, pandas as pd
import streamlit as st, pandas as pd
import pymupdf as fitz
from src.ingest import parse_pdf
from src.retrieve import Index
from src.generate import answer
from src.eval import evaluate_retrieval, evaluate_thresholds, is_llm_error, load_questions

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_case(pdf_paths, case_name):
    chunks, stats, paths = [], [], {}
    for name, path in pdf_paths:
        parsed, page_stats = parse_pdf(path, name)
        chunks.extend(parsed)
        paths[name] = path
        stats.append(dict(
            document=name,
            pages=len(page_stats),
            sections_chunks=len(parsed),
            tables=sum(page["tables"] for page in page_stats),
            scanned_pages=", ".join(
                str(page["page"]) for page in page_stats if page["scanned"]
            ) or "none",
        ))
    return {"index": Index(chunks), "stats": stats, "paths": paths, "case": case_name}


def clear_evaluation():
    st.session_state.pop("retrieval_evaluation", None)
    st.session_state.pop("llm_evaluation", None)

st.set_page_config(layout="wide", page_title="LexiSyncAI")
st.title("LexiSyncAI: Evidence-Grounded Legal Document Analyst")
case = st.sidebar.text_input("Case name", "My case")
files = st.sidebar.file_uploader("Upload 2-5 public legal PDFs", type="pdf", accept_multiple_files=True)
multihop = st.sidebar.checkbox("Multi-hop query decomposition", True)

if st.sidebar.button("Index documents"):
    if not files or not 2 <= len(files) <= 5:
        st.sidebar.error("Please upload between 2 and 5 PDFs.")
    else:
        with st.spinner("Parsing and indexing..."):
            pdf_paths = []
            for f in files:
                p = os.path.join(tempfile.mkdtemp(), f.name)
                with open(p, "wb") as target:
                    target.write(f.getbuffer())
                pdf_paths.append((f.name, p))
            st.session_state.update(build_case(pdf_paths, case))
            clear_evaluation()

if st.sidebar.button("Use bundled benchmark PDFs"):
    data_dir = os.path.join(PROJECT_ROOT, "data")
    benchmark_paths = [
        (name, os.path.join(data_dir, name))
        for name in ("gdpr.pdf", "contract.pdf", "act.pdf")
    ]
    with st.spinner("Indexing bundled benchmark PDFs..."):
        st.session_state.update(build_case(benchmark_paths, "Bundled legal benchmark"))
        clear_evaluation()

if "index" in st.session_state:
    st.subheader(f"Case: {st.session_state.case} — detected structure")
    st.dataframe(pd.DataFrame(st.session_state.stats), use_container_width=True)
    q = st.text_input("Ask a question about the case")
    if q:
        with st.spinner("Retrieving, reranking, generating..."):
            r = answer(q, st.session_state.index, multihop=multihop)
        L, R = st.columns(2)
        col = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red", "INSUFFICIENT EVIDENCE": "gray"}[r["confidence"]]
        L.markdown(f"### Confidence: :{col}[{r['confidence']}]")
        if r["confidence"] == "INSUFFICIENT EVIDENCE":
            L.warning("The documents do not contain enough evidence to answer this.")
            L.write(f"**Searched {r['n_chunks']} passages in:** {', '.join(r['searched'])}")
            if r["out"].get("explanation"):
                L.caption(r["out"]["explanation"])
        else:
            for c in r["out"]["claims"]:
                L.write(f"- {c['text']} **[{', '.join(c['cites'])}]**")
            if r["out"].get("conflicts"):
                L.info("⚠️ Conflicts between sources: " + r["out"]["conflicts"])
            L.caption("Why this evidence: " + str(r["out"].get("explanation", "")))
        L.caption(f"Sub-queries: {r['subqueries']} | tokens: {r['tokens']} | "
                  + " | ".join(f"{k}: {v:.1f}s" for k, v in r["timings"].items()))
        R.subheader("Cited evidence (page highlighted)")
        for k, c in r["ctx"].items():
            with R.expander(f"[{k}] {c['doc']} — p.{c['page']} — {c['clause']}"):
                st.write(c["text"])
                doc = fitz.open(st.session_state.paths[c["doc"]])
                pg = doc[c["page"] - 1]
                for (pn, x0, y0, x1, y1) in c["bbox"]:
                    if pn == c["page"]:
                        pg.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(1, .75, 0), width=2)
                st.image(pg.get_pixmap(dpi=90).tobytes("png"))
else:
    st.info("Create a case: upload 2-5 PDFs in the sidebar and click **Index documents**.")

st.divider()
st.header("Evaluation")
st.caption("Recall uses the gold keywords/pages in src/questions.jsonl. Retrieval metrics run without any LLM calls.")

if "index" not in st.session_state:
    st.info("Index your PDFs, or choose **Use bundled benchmark PDFs** in the sidebar.")
elif st.button("Run retrieval evaluation"):
    questions = load_questions()
    with st.spinner("Running retrieval-only evaluation..."):
        st.session_state.retrieval_evaluation = evaluate_retrieval(
            st.session_state.index, questions
        )
        st.session_state.pop("llm_evaluation", None)

if "retrieval_evaluation" in st.session_state:
    retrieval = st.session_state.retrieval_evaluation
    reranked = next(
        row for row in retrieval["rows"]
        if row["Configuration"] == "Hybrid + cross-encoder rerank"
    )
    llm_evaluation = st.session_state.get("llm_evaluation")
    a, b, c, d = st.columns(4)
    a.metric("Questions used", retrieval["questions_used"])
    b.metric("Recall@5 (hybrid + rerank)", f"{reranked['Recall@5']:.1%}")
    c.metric(
        "Mean latency (hybrid + rerank)",
        f"{reranked['Latency (ms)']:.0f} ms",
    )
    d.metric(
        "Abstention accuracy",
        f"{llm_evaluation['abstention_accuracy']:.1%}"
        if llm_evaluation and llm_evaluation["abstention_accuracy"] is not None
        else "Not run",
    )

    if retrieval["gold_coverage"] < 1:
        st.warning(
            f"Only {retrieval['gold_covered_questions']} of "
            f"{retrieval['answerable_questions']} answerable benchmark questions have "
            "their gold evidence anywhere in the indexed PDFs. Recall cannot exceed "
            "that coverage. For the bundled benchmark, use gdpr.pdf, contract.pdf, and act.pdf."
        )

    st.subheader("Recall@k per configuration")
    st.dataframe(
        pd.DataFrame(retrieval["rows"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Recall@1": st.column_config.NumberColumn(format="%.3f"),
            "Recall@3": st.column_config.NumberColumn(format="%.3f"),
            "Recall@5": st.column_config.NumberColumn(format="%.3f"),
            "Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    if st.button("Run LLM abstention calibration"):
        questions = load_questions()
        with st.spinner("Testing the LLM once per question; retrieval metrics are already saved..."):
            pipeline_results, pipeline_latencies = [], []
            for item in questions:
                started = time.perf_counter()
                pipeline_results.append(
                    answer(item["q"], st.session_state.index, multihop=False)
                )
                pipeline_latencies.append((time.perf_counter() - started) * 1000)
            valid_pairs = [
                (question, result)
                for question, result in zip(questions, pipeline_results)
                if not is_llm_error(result)
            ]
            valid_unanswerable = [
                result for question, result in valid_pairs if not question["answerable"]
            ]
            abstention_accuracy = (
                sum(r["confidence"] == "INSUFFICIENT EVIDENCE" for r in valid_unanswerable)
                / len(valid_unanswerable)
                if valid_unanswerable else None
            )
            st.session_state.llm_evaluation = {
                "pipeline_errors": [
                    result["out"].get("explanation", "")
                    for result in pipeline_results if is_llm_error(result)
                ],
                "successful_questions": len(valid_pairs),
                "abstention_accuracy": abstention_accuracy,
                "pipeline_latencies": pipeline_latencies,
                "threshold_sweep": evaluate_thresholds(questions, pipeline_results)
                if valid_pairs else [],
            }
            llm_evaluation = st.session_state.llm_evaluation

    if llm_evaluation:
        if llm_evaluation["pipeline_errors"]:
            st.warning(
                f"The LLM failed for {len(llm_evaluation['pipeline_errors'])} question(s). "
                f"Abstention metrics exclude failures; {llm_evaluation['successful_questions']} "
                "questions were evaluated successfully."
            )
            with st.expander("LLM error details"):
                api_key = os.getenv("LLM_API_KEY", "")
                for detail in dict.fromkeys(llm_evaluation["pipeline_errors"]):
                    if api_key:
                        detail = detail.replace(api_key, "[REDACTED]")
                    st.code(detail)
        elif not llm_evaluation["successful_questions"]:
            st.warning("No LLM responses succeeded; retrieval metrics remain valid.")

        if llm_evaluation["threshold_sweep"]:
            st.subheader("Abstention-gate calibration (threshold sweep)")
            st.caption(
                f"Computed from {llm_evaluation['successful_questions']} successful LLM responses; "
                "failed requests are excluded."
            )
            st.dataframe(
                pd.DataFrame(llm_evaluation["threshold_sweep"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "threshold": st.column_config.NumberColumn(format="%.2f"),
                    "abstention_accuracy": st.column_config.NumberColumn(format="%.3f"),
                    "false_abstain_rate": st.column_config.NumberColumn(format="%.3f"),
                },
            )

    st.caption(
        "LLM calibration is optional and separate. It makes one answer-generation request "
        "per question; retrieval recall and latency never depend on the LLM."
    )

st.divider()
st.header("Architecture")
diagram_path = os.path.join(PROJECT_ROOT, "docs", "architecture.svg")
with open(diagram_path, "rb") as diagram:
    encoded_diagram = base64.b64encode(diagram.read()).decode("ascii")
st.markdown(
    f'<img src="data:image/svg+xml;base64,{encoded_diagram}" alt="LexiSyncAI architecture diagram" '
    'style="width:100%;height:auto;">',
    unsafe_allow_html=True,
)
