"""
VEDA — Autonomous Data Science System
agents/nlp/summarization.py — Text Summarization Agent

Summarizes text using:
- Extractive summarization (TF-IDF sentence ranking)
- Abstractive summarization via Groq LLM
- Batch summarization for large datasets
"""

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class TextSummarizationAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="TextSummarizationAgent",
            domain="nlp",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _load_text_data(self):
        """Load text data."""
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_cleaned_text.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
            if not files:
                return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """Extract top N most important sentences using TF-IDF scoring."""
        if not isinstance(text, str) or len(text.split()) < 10:
            return text

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= num_sentences:
            return text

        # Score sentences by word frequency
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        sentence_scores = []
        for sent in sentences:
            score = sum(word_freq.get(w.lower(), 0) for w in sent.split())
            sentence_scores.append(score)

        top_indices = sorted(
            range(len(sentence_scores)),
            key=lambda i: sentence_scores[i],
            reverse=True
        )[:num_sentences]

        top_indices = sorted(top_indices)
        summary = ". ".join([sentences[i] for i in top_indices])
        return summary + "."

    def _abstractive_summary(self, texts: list, context: str = "") -> str:
        """Use Groq to generate abstractive summary of text collection."""
        combined = " | ".join([str(t)[:200] for t in texts[:20]])

        prompt = """Summarize these texts in 3-5 sentences. Be specific and factual.

Context: """ + context + """

Texts:
""" + combined + """

Write a concise summary:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert text summarizer. Be concise and factual."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.log("Abstractive summary failed: " + str(e), level="WARN")
            return "Summary generation failed."

    def _compute_text_metrics(self, original: pd.Series, summary: pd.Series) -> dict:
        """Compare original vs summary lengths."""
        orig_len = original.fillna("").astype(str).str.split().str.len().mean()
        summ_len = summary.fillna("").astype(str).str.split().str.len().mean()
        compression_ratio = round(float(summ_len / orig_len) if orig_len > 0 else 0, 4)
        return {
            "avg_original_words": round(float(orig_len), 2),
            "avg_summary_words": round(float(summ_len), 2),
            "compression_ratio": compression_ratio
        }

    def run(self, state: dict) -> dict:
        """
        Text Summarization:
        1. Load text data
        2. Extractive summarization on all texts
        3. Abstractive summarization of dataset
        4. Save summaries
        """

        self.log("Loading text data for summarization...")
        df = self._load_text_data()

        if df is None:
            self.log("No text data found", level="WARN")
            return state

        text_col = "cleaned_text" if "cleaned_text" in df.columns else df.select_dtypes(include="object").columns[0]
        self.log("Summarizing column: " + text_col)
        self.log("Total texts: " + str(len(df)))

        # Extractive summarization on all texts
        self.log("Running extractive summarization...")
        df["extractive_summary"] = df[text_col].apply(
            lambda x: self._extractive_summary(str(x), num_sentences=2)
        )

        # Compute compression metrics
        metrics = self._compute_text_metrics(df[text_col], df["extractive_summary"])
        self.log("Compression metrics: " + str(metrics))

        # Abstractive summary of entire dataset
        self.log("Generating abstractive summary of dataset with Groq...")
        goal = state.get("goal", "")
        sample_texts = df[text_col].dropna().head(20).tolist()
        dataset_summary = self._abstractive_summary(sample_texts, context=goal)
        self.log("Dataset summary generated")
        self.log("Summary: " + dataset_summary[:200] + "...")

        # Sample summaries
        sample_summaries = []
        for i, (orig, summ) in enumerate(zip(
            df[text_col].head(5).tolist(),
            df["extractive_summary"].head(5).tolist()
        )):
            sample_summaries.append({
                "original": str(orig)[:200],
                "summary": str(summ)[:200]
            })

        # Save results
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        summarization_results = {
            "text_column": text_col,
            "total_texts": len(df),
            "compression_metrics": metrics,
            "dataset_abstractive_summary": dataset_summary,
            "sample_summaries": sample_summaries
        }

        results_path = "outputs/" + run_id + "_summarization_results.json"
        with open(results_path, "w") as f:
            json.dump(summarization_results, f, indent=2, default=str)

        # Save enriched dataframe with summaries
        enriched_path = "outputs/" + run_id + "_with_summaries.parquet"
        df.to_parquet(enriched_path, index=False)

        state["summarization_results"] = summarization_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] TextSummarizationAgent: " +
            str(len(df)) + " texts summarized, compression=" +
            str(metrics["compression_ratio"])
        )

        self.log("=" * 50)
        self.log("SUMMARIZATION COMPLETE")
        self.log("Texts processed : " + str(len(df)))
        self.log("Compression     : " + str(metrics["compression_ratio"]))
        self.log("Dataset summary : " + dataset_summary[:150] + "...")
        self.log("=" * 50)

        return state