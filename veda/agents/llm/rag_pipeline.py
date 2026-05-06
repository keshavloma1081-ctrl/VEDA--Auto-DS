"""
VEDA — Autonomous Data Science System
agents/llm/rag_pipeline.py — RAG Pipeline Agent

Retrieval Augmented Generation:
- Retrieves relevant context from vector DB
- Augments LLM prompts with retrieved context
- Answers questions about the dataset
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class RAGPipelineAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="RAGPipelineAgent",
            domain="llm",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _load_vector_db(self, state: dict):
        """Load FAISS index and texts."""
        d = "outputs"

        # Load index
        index_files = [f for f in os.listdir(d) if f.endswith("_faiss.index")]
        if not index_files:
            return None, None

        try:
            import faiss
            index_path = os.path.join(d, sorted(index_files)[-1])
            index = faiss.read_index(index_path)
        except Exception as e:
            self.log("Could not load FAISS index: " + str(e), level="WARN")
            return None, None

        # Load texts
        text_files = [f for f in os.listdir(d) if f.endswith("_cleaned_text.parquet")]
        if not text_files:
            text_files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        if not text_files:
            return index, []

        df = pd.read_parquet(os.path.join(d, sorted(text_files)[-1]))
        text_col = "cleaned_text" if "cleaned_text" in df.columns else df.select_dtypes(include="object").columns[0]
        texts = df[text_col].fillna("").astype(str).tolist()[:1000]

        return index, texts

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode([text])[0]
        except:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=256)
            matrix = vec.fit_transform([text, "placeholder"])
            return matrix.toarray()[0].astype(np.float32)

    def _retrieve(self, index, query_embedding: np.ndarray,
                  texts: list, top_k: int = 3) -> list:
        """Retrieve top-k relevant texts."""
        import faiss
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = index.search(query, top_k)
        retrieved = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(texts):
                retrieved.append(texts[idx])
        return retrieved

    def _generate_answer(self, question: str, context: list) -> str:
        """Generate answer using retrieved context."""
        context_str = "\n".join(["- " + str(c)[:200] for c in context])
        prompt = """Answer this question using only the provided context.

Question: """ + question + """

Context:
""" + context_str + """

Answer in 2-3 sentences. If context is insufficient, say so."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are VEDA. Answer based only on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "RAG answer generation failed: " + str(e)

    def run(self, state: dict) -> dict:
        """
        RAG Pipeline:
        1. Load vector DB
        2. Define questions about the dataset
        3. Retrieve relevant context for each question
        4. Generate answers
        5. Save Q&A results
        """

        self.log("Loading vector DB...")
        index, texts = self._load_vector_db(state)

        if index is None or not texts:
            self.log("Vector DB not available — running without retrieval", level="WARN")
            # Fallback: use pipeline logs as context
            texts = state.get("planner_decision_log", [])
            if not texts:
                state.setdefault("planner_decision_log", []).append(
                    "[" + datetime.now().isoformat() + "] RAGPipelineAgent: skipped — no vector DB"
                )
                return state

        self.log("Vector DB loaded: " + str(len(texts)) + " texts")

        # Define questions
        goal = state.get("goal", "")
        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column", "") if data_profile else ""

        questions = [
            "What are the main patterns in this dataset?",
            "What factors most influence " + str(target_col) + "?",
            "What data quality issues were found?",
            "What actionable insights does this data provide?"
        ]

        qa_results = []
        for question in questions:
            self.log("Q: " + question)

            # Retrieve context
            if index is not None:
                try:
                    query_embedding = self._get_embedding(question)
                    context = self._retrieve(index, query_embedding, texts, top_k=3)
                except Exception as e:
                    context = texts[:3]
            else:
                context = texts[:3]

            # Generate answer
            answer = self._generate_answer(question, context)
            self.log("A: " + answer[:100] + "...")

            qa_results.append({
                "question": question,
                "context_used": [str(c)[:100] for c in context],
                "answer": answer
            })

        # Save results
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        rag_path = "outputs/" + run_id + "_rag_results.json"
        with open(rag_path, "w") as f:
            json.dump(qa_results, f, indent=2)

        state["rag_results"] = qa_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] RAGPipelineAgent: " +
            str(len(qa_results)) + " questions answered"
        )

        self.log("=" * 50)
        self.log("RAG PIPELINE COMPLETE")
        self.log(str(len(qa_results)) + " questions answered with context")
        self.log("=" * 50)

        return state