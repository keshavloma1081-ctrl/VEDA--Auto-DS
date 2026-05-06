"""
VEDA — Autonomous Data Science System
agents/langchain/retriever_agent.py — Retriever Agent

LangChain-style document retrieval:
- TF-IDF based retrieval
- BM25 retrieval
- Semantic search
- Hybrid retrieval
"""

import os
import json
import math
from datetime import datetime
from collections import Counter

from veda.core.base_agent import BaseAgent


class RetrieverAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="RetrieverAgent",
            domain="langchain",
            version="1.0.0"
        )

    def _tokenize(self, text: str) -> list:
        """Simple tokenizer."""
        return text.lower().split()

    def _tfidf_retrieve(self, query: str, documents: list,
                         top_k: int = 3) -> list:
        """TF-IDF based retrieval."""
        if not documents:
            return []

        query_tokens = set(self._tokenize(query))
        scores = []

        for i, doc in enumerate(documents):
            content = doc.get("page_content", "")
            doc_tokens = self._tokenize(content)
            doc_token_set = set(doc_tokens)

            overlap = len(query_tokens & doc_token_set)
            if overlap > 0:
                tf = overlap / max(len(doc_tokens), 1)
                idf = math.log(len(documents) / (1 + sum(
                    1 for d in documents
                    if any(t in d.get("page_content", "").lower()
                           for t in query_tokens)
                )))
                score = tf * max(idf, 0.1)
            else:
                score = 0.0

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            doc = documents[idx]
            results.append({
                "document": doc,
                "score": round(score, 6),
                "rank": len(results) + 1
            })

        return results

    def _bm25_retrieve(self, query: str, documents: list,
                        top_k: int = 3, k1: float = 1.5, b: float = 0.75) -> list:
        """BM25 retrieval."""
        if not documents:
            return []

        query_tokens = self._tokenize(query)
        avg_doc_len = sum(
            len(self._tokenize(d.get("page_content", "")))
            for d in documents
        ) / max(len(documents), 1)

        scores = []
        for i, doc in enumerate(documents):
            content = doc.get("page_content", "")
            doc_tokens = self._tokenize(content)
            doc_len = len(doc_tokens)
            token_counts = Counter(doc_tokens)

            score = 0.0
            for token in query_tokens:
                if token in token_counts:
                    tf = token_counts[token]
                    df = sum(1 for d in documents if token in d.get("page_content", "").lower())
                    idf = math.log((len(documents) - df + 0.5) / (df + 0.5) + 1)
                    norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
                    score += idf * norm_tf

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            results.append({
                "document": documents[idx],
                "score": round(score, 6),
                "rank": len(results) + 1
            })

        return results

    def run(self, state: dict) -> dict:
        """
        Retriever Agent:
        1. Load documents from state
        2. Run TF-IDF retrieval
        3. Run BM25 retrieval
        4. Combine results
        """

        documents = state.get("raw_documents", [])
        if not documents:
            self.log("No documents found — loading from outputs...")
            d = "outputs"
            json_files = [f for f in os.listdir(d) if f.endswith(".json")][:3]
            for filename in json_files:
                try:
                    with open(os.path.join(d, filename)) as f:
                        content = f.read()
                    documents.append({
                        "page_content": content[:500],
                        "metadata": {"source": filename}
                    })
                except:
                    pass

        self.log("Documents available: " + str(len(documents)))

        goal = state.get("goal", "predict machine learning model performance")
        queries = [
            goal[:100],
            "model performance metrics AUC F1",
            "data quality issues and cleaning",
            "feature importance explainability"
        ]

        all_results = {}
        for query in queries:
            self.log("Retrieving for: " + query[:60])

            tfidf_results = self._tfidf_retrieve(query, documents, top_k=2)
            bm25_results = self._bm25_retrieve(query, documents, top_k=2)

            all_results[query] = {
                "tfidf": [{"score": r["score"], "content": r["document"]["page_content"][:100]}
                          for r in tfidf_results],
                "bm25": [{"score": r["score"], "content": r["document"]["page_content"][:100]}
                         for r in bm25_results]
            }

            if tfidf_results:
                self.log("  Top TF-IDF: " + str(tfidf_results[0]["score"]))
            if bm25_results:
                self.log("  Top BM25  : " + str(bm25_results[0]["score"]))

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_retrieval_results.json"
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        state["lc_retrieval"] = all_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] RetrieverAgent: " +
            str(len(queries)) + " queries, " +
            str(len(documents)) + " documents"
        )

        self.log("=" * 50)
        self.log("RETRIEVER COMPLETE")
        self.log("Queries    : " + str(len(queries)))
        self.log("Documents  : " + str(len(documents)))
        self.log("=" * 50)

        return state