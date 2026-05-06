"""
VEDA — Autonomous Data Science System
agents/llm/vector_db.py — Vector Database Agent

Builds a FAISS vector store from text data.
Enables semantic search over dataset.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from veda.core.base_agent import BaseAgent

load_dotenv()


class VectorDBAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="VectorDBAgent",
            domain="llm",
            version="1.0.0"
        )

    def _load_text_data(self, state: dict):
        """Load text data from outputs."""
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_cleaned_text.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        if not files:
            return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _get_embeddings(self, texts: list) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            self.log("Loading sentence transformer model...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self.log("Generating embeddings for " + str(len(texts)) + " texts...")
            embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
            return embeddings
        except Exception as e:
            self.log("Sentence transformer failed: " + str(e), level="WARN")
            self.log("Falling back to TF-IDF embeddings...")
            return self._tfidf_embeddings(texts)

    def _tfidf_embeddings(self, texts: list) -> np.ndarray:
        """Fallback TF-IDF based embeddings."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=256)
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray().astype(np.float32)

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings."""
        import faiss
        dimension = embeddings.shape[1]
        embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build flat index (exact search)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        self.log("FAISS index built: " + str(index.ntotal) + " vectors, dim=" + str(dimension))
        return index

    def _semantic_search(self, index, query_embedding: np.ndarray,
                         texts: list, top_k: int = 5) -> list:
        """Search for most similar texts."""
        import faiss
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(texts):
                results.append({
                    "text": str(texts[idx])[:200],
                    "similarity": round(float(dist), 4),
                    "index": int(idx)
                })
        return results

    def run(self, state: dict) -> dict:
        """
        Vector DB Pipeline:
        1. Load text data
        2. Generate embeddings
        3. Build FAISS index
        4. Demo semantic search
        5. Save index
        """

        self.log("Loading text data...")
        df = self._load_text_data(state)

        if df is None:
            self.log("No text data found — creating from pipeline context", level="WARN")
            # Create text from pipeline context
            goal = state.get("goal", "")
            logs = state.get("planner_decision_log", [])
            texts = [goal] + logs[:20]
            df = pd.DataFrame({"text": texts})

        text_col = "cleaned_text" if "cleaned_text" in df.columns else df.select_dtypes(include="object").columns[0]
        texts = df[text_col].fillna("").astype(str).tolist()

        # Limit for speed
        max_texts = min(len(texts), 1000)
        texts = texts[:max_texts]
        self.log("Building vector DB for " + str(len(texts)) + " texts...")

        # Generate embeddings
        embeddings = self._get_embeddings(texts)
        self.log("Embeddings shape: " + str(embeddings.shape))

        # Build FAISS index
        self.log("Building FAISS index...")
        index = self._build_faiss_index(embeddings)

        # Demo search
        self.log("Running demo semantic search...")
        goal = state.get("goal", "predict")
        query_texts = [goal[:100]]
        query_embeddings = self._get_embeddings(query_texts)
        search_results = self._semantic_search(index, query_embeddings[0], texts, top_k=3)

        self.log("Top 3 similar texts to goal:")
        for r in search_results:
            self.log("  [" + str(r["similarity"]) + "] " + r["text"][:80])

        # Save index
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        try:
            import faiss
            index_path = "outputs/" + run_id + "_faiss.index"
            faiss.write_index(index, index_path)
            self.log("FAISS index saved to: " + index_path)
        except Exception as e:
            self.log("Could not save index: " + str(e), level="WARN")
            index_path = None

        # Save metadata
        metadata = {
            "num_vectors": len(texts),
            "embedding_dim": embeddings.shape[1],
            "text_column": text_col,
            "demo_search_results": search_results,
            "index_path": index_path
        }

        meta_path = "outputs/" + run_id + "_vector_db.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        state["vector_db"] = metadata
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] VectorDBAgent: " +
            str(len(texts)) + " vectors indexed, dim=" + str(embeddings.shape[1])
        )

        self.log("=" * 50)
        self.log("VECTOR DB COMPLETE")
        self.log("Vectors   : " + str(len(texts)))
        self.log("Dimension : " + str(embeddings.shape[1]))
        self.log("=" * 50)

        return state