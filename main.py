import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from typing import List
from . import ingest, retriever

app = FastAPI()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class IngestRequest(BaseModel):
    pdf_folder: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.on_event('startup')
def startup():
    # warm models
    app.state.retriever = retriever.Retriever()


@app.post('/ingest')
def ingest_endpoint(req: IngestRequest):
    if not os.path.isdir(req.pdf_folder):
        raise HTTPException(status_code=400, detail='pdf_folder not found')
    ingest.ingest_folder(req.pdf_folder)
    return {"status": "ok"}


@app.post('/query')
def query_endpoint(req: QueryRequest):
    r = app.state.retriever
    results = r.query(req.question, top_k=req.top_k)
    context = "\n\n".join([f"[source={it['source']}#p{it['page']}#c{it['chunk_index']}]\n{it['text']}" for it in results])

    system = "Use only the provided context to answer the question. Provide a concise answer and list the sources used."
    prompt = f"Context:\n{context}\n\nQuestion: {req.question}\n\nAnswer concisely and include sources."

    if not openai.api_key:
        raise HTTPException(status_code=500, detail='OPENAI_API_KEY not set')

    resp = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":prompt},
        ],
        temperature=0,
        max_tokens=512,
    )

    answer = resp['choices'][0]['message']['content']
    return {"answer": answer, "retrieved": results}
