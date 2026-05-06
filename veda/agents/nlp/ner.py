"""
VEDA — Autonomous Data Science System
agents/nlp/ner.py — Named Entity Recognition Agent

Extracts named entities from text:
- PERSON, ORG, GPE, DATE, MONEY
- Custom entity patterns
- Entity frequency analysis
- Entity features for downstream ML
"""

import os
import json
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from collections import Counter

from veda.core.base_agent import BaseAgent

load_dotenv()


class NERAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="NERAgent",
            domain="nlp",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _load_cleaned_text(self):
        """Load cleaned text from preprocessing step."""
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_cleaned_text.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
            if not files:
                return None
        return pd.read_parquet(os.path.join(d, sorted(files)[-1]))

    def _rule_based_ner(self, text: str) -> dict:
        """Simple rule-based NER using regex patterns."""
        entities = {
            "MONEY": [],
            "PERCENT": [],
            "DATE": [],
            "EMAIL": [],
            "URL": [],
            "PHONE": []
        }

        # Money patterns
        money_pattern = r"\$[\d,]+\.?\d*|\d+[\s]?(dollars?|rupees?|USD|INR|EUR)"
        entities["MONEY"] = re.findall(money_pattern, text, re.IGNORECASE)

        # Percent patterns
        percent_pattern = r"\d+\.?\d*\s?%"
        entities["PERCENT"] = re.findall(percent_pattern, text)

        # Date patterns
        date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4}\b"
        entities["DATE"] = re.findall(date_pattern, text, re.IGNORECASE)

        # Email patterns
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        entities["EMAIL"] = re.findall(email_pattern, text)

        # URL patterns
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        entities["URL"] = re.findall(url_pattern, text)

        return {k: list(set(v)) for k, v in entities.items() if v}

    def _llm_ner(self, texts: list) -> list:
        """Use Groq LLM for NER on sample texts."""
        all_entities = []

        for text in texts[:10]:  # Process 10 samples
            prompt = """Extract named entities from this text. Return JSON only.

Text: """ + str(text)[:500] + """

Return format:
{"PERSON": [], "ORG": [], "LOCATION": [], "PRODUCT": [], "EVENT": []}

Only include entities that actually appear in the text. Return valid JSON only."""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an NER system. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=200
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
                entities = json.loads(raw)
                all_entities.append(entities)
            except Exception as e:
                all_entities.append({})

        return all_entities

    def _aggregate_entities(self, entity_list: list) -> dict:
        """Aggregate entities across all samples."""
        aggregated = {}
        for entities in entity_list:
            for entity_type, values in entities.items():
                if entity_type not in aggregated:
                    aggregated[entity_type] = []
                aggregated[entity_type].extend(values)

        # Count and deduplicate
        result = {}
        for entity_type, values in aggregated.items():
            counter = Counter(values)
            result[entity_type] = dict(counter.most_common(10))
        return result

    def _create_entity_features(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Create numeric features from entity presence."""
        patterns = {
            "has_money": r"\$[\d,]+\.?\d*|\d+[\s]?(dollars?|rupees?|USD|INR|EUR)",
            "has_percent": r"\d+\.?\d*\s?%",
            "has_date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "has_email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "has_url": r"http[s]?://",
            "entity_count": None
        }

        texts = df[text_col].fillna("").astype(str)

        for feature, pattern in patterns.items():
            if pattern:
                df[feature] = texts.str.contains(pattern, regex=True, case=False).astype(int)
            else:
                # Count total entity mentions
                all_patterns = [p for p in patterns.values() if p]
                counts = sum(texts.str.count(p) for p in all_patterns)
                df[feature] = counts

        return df

    def run(self, state: dict) -> dict:
        """
        NER Pipeline:
        1. Load cleaned text
        2. Rule-based NER on all samples
        3. LLM-based NER on sample
        4. Aggregate entity findings
        5. Create entity features
        """

        self.log("Loading cleaned text for NER...")
        df = self._load_cleaned_text()

        if df is None:
            self.log("No text data found for NER", level="WARN")
            return state

        # Find text column
        text_col = "cleaned_text" if "cleaned_text" in df.columns else df.select_dtypes(include="object").columns[0]
        self.log("Running NER on column: " + text_col)
        self.log("Total texts: " + str(len(df)))

        # Rule-based NER on full dataset
        self.log("Running rule-based NER on full dataset...")
        all_rule_entities = []
        for text in df[text_col].fillna("").astype(str):
            entities = self._rule_based_ner(text)
            all_rule_entities.append(entities)

        # Aggregate rule-based entities
        rule_aggregated = self._aggregate_entities(all_rule_entities)
        self.log("Rule-based entities found: " + str(list(rule_aggregated.keys())))

        # LLM-based NER on sample
        self.log("Running LLM-based NER on 10 sample texts...")
        sample_texts = df[text_col].dropna().head(10).tolist()
        llm_entities = self._llm_ner(sample_texts)
        llm_aggregated = self._aggregate_entities(llm_entities)
        self.log("LLM entities found: " + str(list(llm_aggregated.keys())))

        # Create entity features
        self.log("Creating entity features...")
        df = self._create_entity_features(df, text_col)

        # Save results
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        ner_results = {
            "text_column": text_col,
            "total_texts": len(df),
            "rule_based_entities": rule_aggregated,
            "llm_entities": llm_aggregated,
            "entity_features_created": ["has_money", "has_percent", "has_date",
                                         "has_email", "has_url", "entity_count"]
        }

        ner_path = "outputs/" + run_id + "_ner_results.json"
        with open(ner_path, "w") as f:
            json.dump(ner_results, f, indent=2, default=str)

        # Save enriched dataframe
        enriched_path = "outputs/" + run_id + "_ner_features.parquet"
        df.to_parquet(enriched_path, index=False)

        state["ner_results"] = ner_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] NERAgent: entities extracted from " + text_col
        )

        self.log("=" * 50)
        self.log("NER COMPLETE")
        self.log("Rule entities : " + str(list(rule_aggregated.keys())))
        self.log("LLM entities  : " + str(list(llm_aggregated.keys())))
        self.log("Features added: has_money, has_percent, has_date, has_email, has_url")
        self.log("=" * 50)

        return state