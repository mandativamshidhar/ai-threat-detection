import os
import sqlite3
from typing import List, Tuple
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DB_PATH = os.path.join(os.path.dirname(__file__), "metadata.db")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss.index")


def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        text = p.extract_text() or ""
        pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 50) -> List[str]:
    # Chunk by words. Defaults tuned for latency/recall tradeoff.
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - chunk_overlap)
    return chunks


def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        page INTEGER,
        chunk_index INTEGER,
        text TEXT
    )
    """
    )
    conn.commit()
    conn.close()


def ingest_folder(pdf_folder: str, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 512, hnsw_m: int = 32, chunk_size: int = 200, chunk_overlap: int = 50) -> None:
    ensure_db()
    model = SentenceTransformer(model_name)
    dims = model.get_sentence_embedding_dimension()

    # create or load an HNSW index wrapped with IDMap for stable ids
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        # M controls HNSW connectivity; tuned for speed/recall tradeoff
        M = int(hnsw_m)
        hnsw = faiss.IndexHNSWFlat(dims, M)
        # use inner product on normalized vectors
        index = faiss.IndexIDMap(hnsw)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    vecs_batch = []
    ids_batch = []

    def flush_batch():
        nonlocal vecs_batch, ids_batch, index
        if not vecs_batch:
            return
        arr = np.vstack(vecs_batch).astype('float32')
        # normalize L2 for inner-product similarity
        faiss.normalize_L2(arr)
        ids = np.array(ids_batch, dtype='int64')
        index.add_with_ids(arr, ids)
        vecs_batch = []
        ids_batch = []

    for fname in os.listdir(pdf_folder):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(pdf_folder, fname)
        pages = extract_text_from_pdf(path)
        for page_no, page_text in pages:
            chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for ci, chunk in enumerate(chunks):
                cur.execute(
                    "INSERT INTO docs (source, page, chunk_index, text) VALUES (?,?,?,?)",
                    (fname, page_no, ci, chunk),
                )
                rowid = cur.lastrowid
                emb = model.encode(chunk, convert_to_numpy=True)
                vecs_batch.append(emb)
                ids_batch.append(rowid)
                if len(vecs_batch) >= batch_size:
                    flush_batch()

    # flush remainder
    flush_batch()
    conn.commit()
    conn.close()
    # persist index
    faiss.write_index(index, INDEX_PATH)
