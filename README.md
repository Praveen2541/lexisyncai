# LexiSyncAI: Evidence-Grounded Legal Document Analyst

Upload 2-5 public legal PDFs, ask questions, get answers where every claim cites a page and clause,
with a confidence badge and abstention when the documents lack the evidence.

## Run
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then paste your API key (Mac/Linux: cp)
streamlit run app/streamlit_app.py     # or: make run
```
Docker: `docker build -t lexisync . && docker run -p 8501:8501 --env-file .env lexisync`

Other commands: `python src/tune_tau.py data/*.pdf` (tune abstention threshold on dev set),
`python src/eval.py data/*.pdf` (evaluation -> docs/results.md), `pytest -q` (tests).

## Architecture
PDFs -> Parser (reading order, header/footer removal, scanned flag, table count) -> Clause-aware chunker
-> BM25 (+clause-number tokenizer) and Dense (bge-small) -> RRF fusion + clause-reference boost
-> Cross-encoder rerank -> Query decomposition (multi-hop) -> LLM (JSON, cited claims)
-> Citation validator + abstention gate -> UI (answer, badge, side-by-side highlighted page).
Put your diagram in `docs/architecture.png`.

## Design justifications
- **Clause-aware chunking**: fixed-size chunks split clause 7.2 from its carve-outs and lose the clause ID, so
  page+clause citations break. One legal unit per chunk, word cap, paragraph fallback for unseen PDFs.
- **BM25 + dense + RRF**: BM25 matches exact clause numbers ("7.2", "Article 83(5)"); dense matches paraphrases;
  RRF merges ranks without score calibration.
- **Cross-encoder**: scores (query, passage) jointly, more accurate than bi-encoder for the final top-k.
- **Grounding**: LLM answers only from numbered context; code rejects claims citing unknown IDs (-> LOW).
- **Abstention**: three signals: cross-encoder top score < TAU_LOW, LLM `answerable=false`, no valid citations.
  TAU_LOW tuned on a separate dev split (`src/dev_questions.jsonl`).
- **Conflicts**: prompt requires both values, sources/pages, and which is used and why.

## Results (fill from `docs/results.md` after `python src/eval.py`)
TODO: recall@k table, abstention accuracy, citation precision, latency, manual correctness table,
no-retrieval vs retrieval comparison.

## Bonus features implemented
Bounding-box highlight of cited passage on page image (#4); latency/token report per stage (#11);
conflict detection across documents (#9); (deployment URL, if done: #10).

## Limitations
No OCR (scanned pages are only flagged); no footnote tagging; table cell extraction not evaluated;
correctness graded manually; eval keywords/pages should be verified against your PDFs.
