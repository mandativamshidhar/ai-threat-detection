import os
import sqlite3
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE, "metadata.db")
INDEX_PATH = os.path.join(BASE, "faiss.index")


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", ef_search: int = 128):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.ef_search = ef_search
        self._ensure_index()

    def _ensure_index(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            # if HNSW, set efSearch
            try:
                if hasattr(self.index, 'hnsw'):
                    self.index.hnsw.efSearch = self.ef_search
                else:
                    # If wrapped in IDMap, underlying index accessible
                    if isinstance(self.index, faiss.IndexIDMap):
                        self.index.index.hnsw.efSearch = self.ef_search
            except Exception:
                pass
        else:
            dims = self.model.get_sentence_embedding_dimension()
            hnsw = faiss.IndexHNSWFlat(dims, 32)
            hnsw.efSearch = self.ef_search
            self.index = faiss.IndexIDMap(hnsw)

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        emb = self.model.encode(text, convert_to_numpy=True)
        vec = np.asarray(emb, dtype='float32')
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        else:
            vec = vec.reshape(1, -1)
        # Ensure vector dim matches index dim; if mismatch, pad or trim (fallback)
        try:
            index_dim = getattr(self.index, 'd', None)
            if index_dim is None and isinstance(self.index, faiss.IndexIDMap):
                index_dim = getattr(self.index.index, 'd', None)
            if index_dim is not None and vec.shape[1] != index_dim:
                if vec.shape[1] > index_dim:
                    vec = vec[:, :index_dim]
                else:
                    pad = np.zeros((1, index_dim - vec.shape[1]), dtype='float32')
                    vec = np.concatenate([vec, pad], axis=1)
        except Exception:
            pass
        faiss.normalize_L2(vec)
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(vec, top_k)
        ids = I[0].tolist()
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        results = []
        for idx, score in zip(ids, D[0].tolist()):
            if idx < 0:
                continue
            # idx is the sqlite row id because we used IndexIDMap with row ids
            rowid = int(idx)
            cur.execute("SELECT id, source, page, chunk_index, text FROM docs WHERE id = ?", (rowid,))
            r = cur.fetchone()
            if r:
                results.append({
                    "id": r[0],
                    "source": r[1],
                    "page": r[2],
                    "chunk_index": r[3],
                    "text": r[4],
                    "score": float(score),
                })
        conn.close()
        return results
