import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st, fitz, pandas as pd
import streamlit as st, pandas as pd
import pymupdf as fitz
from src.ingest import parse_pdf
from src.retrieve import Index
from src.generate import answer

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
            chunks, stats, paths, pagestats = [], [], {}, {}
            for f in files:
                p = os.path.join(tempfile.mkdtemp(), f.name)
                open(p, "wb").write(f.getbuffer())
                c, pg = parse_pdf(p, f.name)
                chunks += c
                paths[f.name] = p
                pagestats[f.name] = pg
                stats.append(dict(document=f.name, pages=len(pg), sections_chunks=len(c),
                                  tables=sum(x["tables"] for x in pg),
                                  scanned_pages=", ".join(str(x["page"]) for x in pg if x["scanned"]) or "none"))
            st.session_state.update(index=Index(chunks), stats=stats, paths=paths, case=case)

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
