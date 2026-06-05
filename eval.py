from typing import List
import sqlite3
import numpy as np
from .retriever import Retriever


def mrr_at_k(queries: List[str], ground_truth_ids: List[int], k: int = 5) -> float:
    r = Retriever()
    scores = []
    for q, gt in zip(queries, ground_truth_ids):
        res = r.query(q, top_k=k)
        ids = [it['id'] for it in res]
        if gt in ids:
            rank = ids.index(gt) + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def ragas_like_faithfulness(answer: str, retrieved_texts: List[str], threshold: float = 0.7) -> float:
    # Simple proxy: measure fraction of answer sentences with a high-similarity match in retrieved docs
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sents = [s.strip() for s in answer.split('.') if s.strip()]
    if not sents:
        return 0.0
    emb_s = model.encode(sents, convert_to_numpy=True)
    doc_embs = model.encode(retrieved_texts, convert_to_numpy=True)
    import numpy as np
    # normalize
    def norm(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / (n + 1e-12)
    emb_s = norm(emb_s)
    doc_embs = norm(doc_embs)
    sim = emb_s @ doc_embs.T
    # for each sentence check if any doc has sim >= threshold
    hits = (sim >= threshold).any(axis=1)
    return float(hits.sum()) / len(sents)
