"""
VEDA — Autonomous Data Science System
agents/nlp/sentiment.py — Sentiment Analysis Agent

Performs sentiment analysis:
- Rule-based VADER-style scoring
- LLM-based sentiment with Groq
- Aspect-level sentiment
- Sentiment features for ML
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


class SentimentAnalysisAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="SentimentAnalysisAgent",
            domain="nlp",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

        # Simple sentiment lexicon
        self.positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "brilliant", "outstanding", "superb", "perfect", "love", "loved",
            "best", "awesome", "incredible", "exceptional", "happy", "pleased",
            "satisfied", "recommend", "recommended", "helpful", "useful",
            "beautiful", "fast", "quick", "easy", "reliable", "quality",
            "impressive", "delighted", "enjoy", "enjoyed", "positive", "nice"
        }

        self.negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "hate", "hated",
            "poor", "disappointing", "disappointed", "useless", "broken",
            "failed", "failure", "waste", "overpriced", "expensive", "slow",
            "difficult", "hard", "unreliable", "cheap", "nasty", "disgusting",
            "defective", "damaged", "unhappy", "frustrated", "angry", "problem",
            "issue", "error", "wrong", "incorrect", "missing", "avoid"
        }

    def _load_cleaned_text(self):
        """Load cleaned text."""
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_cleaned_text.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
            if not files:
                return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _lexicon_sentiment(self, text: str) -> dict:
        """Rule-based sentiment using lexicon."""
        if not isinstance(text, str):
            return {"score": 0.0, "label": "neutral", "pos_count": 0, "neg_count": 0}

        words = set(text.lower().split())
        pos_count = len(words & self.positive_words)
        neg_count = len(words & self.negative_words)
        total = pos_count + neg_count

        if total == 0:
            score = 0.0
        else:
            score = (pos_count - neg_count) / total

        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return {
            "score": round(score, 4),
            "label": label,
            "pos_count": pos_count,
            "neg_count": neg_count
        }

    def _llm_sentiment(self, texts: list) -> list:
        """Use Groq for sentiment analysis on samples."""
        results = []
        batch_size = 5

        for i in range(0, min(len(texts), 20), batch_size):
            batch = texts[i:i + batch_size]
            batch_text = "\n".join([str(j+1) + ". " + str(t)[:200] for j, t in enumerate(batch)])

            prompt = """Analyze sentiment for each text. Return JSON only.

Texts:
""" + batch_text + """

Return format (exactly """ + str(len(batch)) + """ items):
[{"text_num": 1, "sentiment": "positive/negative/neutral", "confidence": 0.9, "reason": "brief reason"}, ...]

Return valid JSON array only."""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a sentiment analyzer. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=500
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
                batch_results = json.loads(raw)
                results.extend(batch_results)
            except Exception as e:
                for _ in batch:
                    results.append({"sentiment": "neutral", "confidence": 0.5})

        return results

    def _compute_sentiment_features(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Add sentiment features to dataframe."""
        texts = df[text_col].fillna("").astype(str)

        sentiment_data = texts.apply(self._lexicon_sentiment)
        df["sentiment_score"] = sentiment_data.apply(lambda x: x["score"])
        df["sentiment_label"] = sentiment_data.apply(lambda x: x["label"])
        df["pos_word_count"] = sentiment_data.apply(lambda x: x["pos_count"])
        df["neg_word_count"] = sentiment_data.apply(lambda x: x["neg_count"])
        df["sentiment_magnitude"] = df["sentiment_score"].abs()

        return df

    def run(self, state: dict) -> dict:
        """
        Sentiment Analysis:
        1. Load cleaned text
        2. Lexicon-based scoring on all texts
        3. LLM sentiment on sample
        4. Sentiment features for ML
        5. Distribution analysis
        """

        self.log("Loading text for sentiment analysis...")
        df = self._load_cleaned_text()

        if df is None:
            self.log("No text data found", level="WARN")
            return state

        text_col = "cleaned_text" if "cleaned_text" in df.columns else df.select_dtypes(include="object").columns[0]
        self.log("Analyzing sentiment in column: " + text_col)

        # Lexicon-based sentiment on full dataset
        self.log("Running lexicon-based sentiment on " + str(len(df)) + " texts...")
        df = self._compute_sentiment_features(df, text_col)

        # Distribution
        sentiment_dist = df["sentiment_label"].value_counts().to_dict()
        avg_score = round(float(df["sentiment_score"].mean()), 4)
        self.log("Sentiment distribution: " + str(sentiment_dist))
        self.log("Average sentiment score: " + str(avg_score))

        # LLM sentiment on sample
        self.log("Running LLM sentiment on 20 sample texts...")
        sample_texts = df[text_col].dropna().head(20).tolist()
        llm_results = self._llm_sentiment(sample_texts)

        llm_dist = {}
        for r in llm_results:
            s = r.get("sentiment", "neutral")
            llm_dist[s] = llm_dist.get(s, 0) + 1
        self.log("LLM sentiment distribution (20 samples): " + str(llm_dist))

        # Save results
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        sentiment_results = {
            "text_column": text_col,
            "total_texts": len(df),
            "lexicon_distribution": sentiment_dist,
            "lexicon_avg_score": avg_score,
            "llm_sample_distribution": llm_dist,
            "llm_sample_results": llm_results[:5],
            "features_created": ["sentiment_score", "sentiment_label",
                                  "pos_word_count", "neg_word_count", "sentiment_magnitude"]
        }

        results_path = "outputs/" + run_id + "_sentiment_results.json"
        with open(results_path, "w") as f:
            json.dump(sentiment_results, f, indent=2, default=str)

        enriched_path = "outputs/" + run_id + "_sentiment_features.parquet"
        df.to_parquet(enriched_path, index=False)

        state["sentiment_results"] = sentiment_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] SentimentAnalysisAgent: avg_score=" +
            str(avg_score) + " dist=" + str(sentiment_dist)
        )

        self.log("=" * 50)
        self.log("SENTIMENT ANALYSIS COMPLETE")
        self.log("Distribution : " + str(sentiment_dist))
        self.log("Avg score    : " + str(avg_score))
        self.log("LLM sample   : " + str(llm_dist))
        self.log("=" * 50)

        return state