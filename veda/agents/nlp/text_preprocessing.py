"""
VEDA — Autonomous Data Science System
agents/nlp/text_preprocessing.py — Text Preprocessing Agent

Handles real NLP preprocessing:
- Tokenization
- Stopword removal
- Lemmatization
- TF-IDF vectorization
- Text cleaning
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from veda.core.base_agent import BaseAgent

load_dotenv()


class TextPreprocessingAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="TextPreprocessingAgent",
            domain="nlp",
            version="1.0.0"
        )
        self.text_col = None
        self.target_col = None

    def _detect_text_columns(self, df: pd.DataFrame) -> list:
        """Detect columns that contain text data."""
        text_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                avg_len = df[col].dropna().astype(str).str.len().mean()
                if avg_len > 20:
                    text_cols.append(col)
        return text_cols

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # Remove emails
        text = re.sub(r"\S+@\S+", "", text)
        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _remove_stopwords(self, text: str, stopwords: set) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered = [w for w in words if w not in stopwords and len(w) > 2]
        return " ".join(filtered)

    def _get_stopwords(self) -> set:
        """Basic English stopwords."""
        return {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "he", "him", "his",
            "she", "her", "hers", "it", "its", "they", "them", "their",
            "what", "which", "who", "whom", "this", "that", "these",
            "those", "am", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "a", "an", "the", "and", "but", "if", "or", "because", "as",
            "until", "while", "of", "at", "by", "for", "with", "about",
            "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in",
            "out", "on", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "now", "also"
        }

    def _vectorize_tfidf(self, texts: pd.Series, max_features: int = 5000):
        """Convert text to TF-IDF features."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            tfidf_matrix = vectorizer.fit_transform(texts.fillna(""))
            feature_names = vectorizer.get_feature_names_out()
            return vectorizer, tfidf_matrix, feature_names
        except Exception as e:
            self.log("TF-IDF failed: " + str(e), level="WARN")
            return None, None, None

    def _compute_text_stats(self, texts: pd.Series) -> dict:
        """Compute basic text statistics."""
        texts_str = texts.fillna("").astype(str)
        return {
            "avg_word_count": round(float(texts_str.str.split().str.len().mean()), 2),
            "max_word_count": int(texts_str.str.split().str.len().max()),
            "min_word_count": int(texts_str.str.split().str.len().min()),
            "avg_char_count": round(float(texts_str.str.len().mean()), 2),
            "empty_texts": int((texts_str == "").sum()),
            "unique_texts": int(texts_str.nunique()),
            "vocabulary_size": len(set(" ".join(texts_str.tolist()).split()))
        }

    def run(self, state: dict) -> dict:
        """
        Main NLP preprocessing:
        1. Detect text columns
        2. Clean text
        3. Remove stopwords
        4. TF-IDF vectorization
        5. Save processed features
        """

        data_profile = state.get("data_profile", {})
        target_col = data_profile.get("target_column") if data_profile else None

        self.log("Loading data for NLP preprocessing...")
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        if not files:
            self.log("No data file found", level="WARN")
            return state
        df = pd.read_parquet(os.path.join(d, sorted(files)[-1]))
        self.log("Loaded " + str(df.shape[0]) + " rows x " + str(df.shape[1]) + " columns")

        # Detect text columns
        text_cols = self._detect_text_columns(df)
        if not text_cols:
            self.log("No text columns detected — skipping NLP preprocessing", level="WARN")
            return state

        self.log("Text columns detected: " + str(text_cols))
        primary_text_col = text_cols[0]
        self.log("Primary text column: " + primary_text_col)

        # Compute raw text stats
        raw_stats = self._compute_text_stats(df[primary_text_col])
        self.log("Raw text stats: " + str(raw_stats))

        # Clean text
        self.log("Cleaning text...")
        stopwords = self._get_stopwords()
        df["cleaned_text"] = df[primary_text_col].apply(self._clean_text)
        df["cleaned_text"] = df["cleaned_text"].apply(
            lambda x: self._remove_stopwords(x, stopwords)
        )

        # Compute cleaned text stats
        cleaned_stats = self._compute_text_stats(df["cleaned_text"])
        self.log("Cleaned text stats: " + str(cleaned_stats))

        # TF-IDF vectorization
        self.log("Vectorizing with TF-IDF (max 5000 features)...")
        vectorizer, tfidf_matrix, feature_names = self._vectorize_tfidf(
            df["cleaned_text"], max_features=5000
        )

        if vectorizer is not None:
            self.log("TF-IDF matrix shape: " + str(tfidf_matrix.shape))
            self.log("Vocabulary size: " + str(len(feature_names)))

            # Convert to DataFrame
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=["tfidf_" + f for f in feature_names]
            )

            # Add target column back
            if target_col and target_col in df.columns:
                tfidf_df[target_col] = df[target_col].values

            # Save TF-IDF features
            os.makedirs("outputs", exist_ok=True)
            run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
            tfidf_path = "outputs/" + run_id + "_tfidf_features.parquet"
            tfidf_df.to_parquet(tfidf_path, index=False)
            self.log("TF-IDF features saved to: " + tfidf_path)

            # Save vectorizer
            vec_path = "outputs/" + run_id + "_tfidf_vectorizer.pkl"
            joblib.dump(vectorizer, vec_path)
            self.log("Vectorizer saved to: " + vec_path)

            # Save cleaned text
            cleaned_path = "outputs/" + run_id + "_cleaned_text.parquet"
            df[["cleaned_text"] + ([target_col] if target_col in df.columns else [])].to_parquet(
                cleaned_path, index=False
            )

            # Top TF-IDF terms
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-20:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            self.log("Top 20 TF-IDF terms: " + str(top_terms))

            # Save stats
            stats_path = "outputs/" + run_id + "_nlp_stats.json"
            nlp_stats = {
                "primary_text_column": primary_text_col,
                "all_text_columns": text_cols,
                "raw_stats": raw_stats,
                "cleaned_stats": cleaned_stats,
                "tfidf_features": int(tfidf_matrix.shape[1]),
                "top_terms": top_terms,
                "vectorizer_path": vec_path,
                "tfidf_path": tfidf_path
            }
            with open(stats_path, "w") as f:
                json.dump(nlp_stats, f, indent=2)

            # Update state
            state["nlp_stats"] = nlp_stats
            state["feature_list"] = ["tfidf_" + f for f in feature_names[:50]]
            state.setdefault("planner_decision_log", []).append(
                "[" + datetime.now().isoformat() + "] TextPreprocessingAgent: " +
                str(tfidf_matrix.shape[1]) + " TF-IDF features created from " +
                primary_text_col
            )

            self.log("=" * 50)
            self.log("NLP PREPROCESSING COMPLETE")
            self.log("Text column    : " + primary_text_col)
            self.log("TF-IDF features: " + str(tfidf_matrix.shape[1]))
            self.log("Top terms      : " + str(top_terms[:10]))
            self.log("=" * 50)

        return state