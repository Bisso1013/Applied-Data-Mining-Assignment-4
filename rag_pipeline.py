import os
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from groq import Groq

load_dotenv()
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class RAGPipeline:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

    def load_data(self):
        dataset = load_dataset("akariasai/PopQA", split="train[:200]")
        self.questions = [x["question"] for x in dataset]
        self.answers = [x["answers"] for x in dataset]

        # simple corpus (use answers as pseudo docs)
        self.corpus = [a[0] if len(a) > 0 else "" for a in self.answers]

    def build_index(self):
        self.embeddings = self.embedder.encode(self.corpus, show_progress_bar=True)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

        tokenized = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    # -------------------------
    # RETRIEVAL
    # -------------------------
    def dense_search(self, query, k=5):
        q_emb = self.embedder.encode([query])
        distances, indices = self.index.search(q_emb, k)
        return indices[0]

    def bm25_search(self, query, k=5):
        scores = self.bm25.get_scores(query.split())
        return np.argsort(scores)[::-1][:k]

    def hybrid_search(self, query, k=5):
        d = self.dense_search(query, k)
        b = self.bm25_search(query, k)

        # reciprocal rank fusion
        scores = {}
        for i, idx in enumerate(d):
            scores[idx] = scores.get(idx, 0) + 1/(i+1)
        for i, idx in enumerate(b):
            scores[idx] = scores.get(idx, 0) + 1/(i+1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranked[:k]]

    def rerank(self, query, indices):
        pairs = [[query, self.corpus[i]] for i in indices]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        return [i for i, _ in ranked]

    # -------------------------
    # QUERY EXPANSION (Groq)
    # -------------------------
    def expand_query(self, query):
        prompt = f"Rewrite this query to improve retrieval:\n{query}"

        res = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()

    # -------------------------
    # GENERATION
    # -------------------------
    def generate_answer(self, query, docs):
        context = "\n".join([f"[P{i}] {doc}" for i, doc in enumerate(docs)])

        prompt = f"""
Answer the question using ONLY the context below.
Cite sources like [P0], [P1].

Context:
{context}

Question: {query}
"""

        res = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

    # -------------------------
    # SELF REFLECTION
    # -------------------------
    def reflect(self, answer):
        prompt = f"Check if this answer is grounded and correct. If not, fix it:\n{answer}"

        res = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

    # -------------------------
    # FULL PIPELINE
    # -------------------------
    def answer_question(self, query):
        expanded = self.expand_query(query)

        indices = self.hybrid_search(expanded)
        indices = self.rerank(query, indices)

        docs = [self.corpus[i] for i in indices]

        answer = self.generate_answer(query, docs)
        final = self.reflect(answer)

        return final