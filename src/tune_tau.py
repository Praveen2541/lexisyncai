
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingest import parse_pdf
from src.retrieve import Index

chunks = []
for f in sys.argv[1:]:
    chunks += parse_pdf(f, os.path.basename(f))[0]
idx = Index(chunks)
D = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "dev_questions.jsonl")) if l.strip()]
a, u = [], []
for x in D:
    s = idx.search(x["q"])[0][1]
    print(f"{'ANS ' if x['answerable'] else 'UNANS'} top={s:.2f}  {x['q']}")
    (a if x["answerable"] else u).append(s)
if a and u:
    print(f"\nlowest answerable = {min(a):.2f}, highest unanswerable = {max(u):.2f}")
    print(f"Suggested TAU_LOW = {(min(a) + max(u)) / 2:.2f}  (set in .env)")
