import os
import time
import argparse
import sqlite3
import statistics
from typing import List

from . import ingest, retriever
from sentence_transformers import SentenceTransformer
import openai


def load_sample_queries(db_path: str, n: int):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM docs ORDER BY RANDOM() LIMIT ?", (n,))
    rows = cur.fetchall()
    conn.close()
    queries = []
    ids = []
    for (rid, t) in rows:
        s = t.strip().split('.')
        if len(s) > 0 and s[0].strip():
            queries.append(s[0].strip())
        else:
            queries.append(t.strip()[:200])
        ids.append(rid)
    return queries, ids


def measure(args):
    # Build or reuse index
    if args.rebuild:
        print(f"Ingesting folder {args.pdf_folder} (M={args.hnsw_m})...")
        ingest.ingest_folder(args.pdf_folder, batch_size=args.batch_size, hnsw_m=args.hnsw_m)

    r = retriever.Retriever(ef_search=args.ef_search)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare queries and ground-truth ids
    queries, gt_ids = load_sample_queries(os.path.join(os.path.dirname(__file__), 'metadata.db'), args.num_queries)

    # Warm-up
    print("Warm-up query...")
    _ = r.query(queries[0], top_k=args.top_k)

    timings = []
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if openai_api_key:
        openai.api_key = openai_api_key

    for q in queries:
        t0 = time.perf_counter()
        t_emb_start = time.perf_counter()
        emb = model.encode(q, convert_to_numpy=True).astype('float32')
        t_emb = time.perf_counter() - t_emb_start

        t_faiss_start = time.perf_counter()
        res = r.query(q, top_k=args.top_k)
        t_faiss = time.perf_counter() - t_faiss_start

        t_openai = 0.0
        if args.mock_openai or not openai_api_key:
            # simulate small network latency if mocking
            t_openai = 0.05
            time.sleep(0.01)
        else:
            t_oa_start = time.perf_counter()
            prompt = "Context:\n\n" + "\n\n".join([it['text'] for it in res]) + f"\n\nQuestion: {q}\n\nAnswer concisely."
            _ = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=128,
            )
            t_openai = time.perf_counter() - t_oa_start

        total = time.perf_counter() - t0
        timings.append({
            'embed': t_emb,
            'faiss': t_faiss,
            'openai': t_openai,
            'total': total,
        })

    totals = [t['total'] for t in timings]
    print(f"Queries: {len(totals)}")
    print(f"Avg total: {statistics.mean(totals):.3f}s, median: {statistics.median(totals):.3f}s, p95: {statistics.quantiles(totals, n=20)[18]:.3f}s")
    print(f"Avg embed: {statistics.mean([t['embed'] for t in timings]):.3f}s")
    print(f"Avg faiss: {statistics.mean([t['faiss'] for t in timings]):.3f}s")
    print(f"Avg openai: {statistics.mean([t['openai'] for t in timings]):.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_folder', required=True)
    parser.add_argument('--num_queries', type=int, default=50)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--ef_search', type=int, default=128)
    parser.add_argument('--hnsw_m', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--chunk_size', type=int, default=200)
    parser.add_argument('--chunk_overlap', type=int, default=50)
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--mock_openai', action='store_true')
    args = parser.parse_args()
    measure(args)


if __name__ == '__main__':
    main()
