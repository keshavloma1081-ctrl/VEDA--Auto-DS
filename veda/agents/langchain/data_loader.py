"""
VEDA — Autonomous Data Science System
agents/langchain/data_loader.py — Data Loader Agent

LangChain-style document loaders:
- CSV loader
- JSON loader
- Text loader
- Parquet loader
- Document chunking
"""

import os
import json
import pandas as pd
from datetime import datetime

from veda.core.base_agent import BaseAgent


class DataLoaderAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="DataLoaderAgent",
            domain="langchain",
            version="1.0.0"
        )

    def _load_document(self, path: str) -> list:
        """Load document and split into chunks."""
        if not os.path.exists(path):
            return []

        ext = path.lower().split(".")[-1]
        documents = []

        if ext == "csv":
            df = pd.read_csv(path, nrows=100)
            for i, row in df.iterrows():
                documents.append({
                    "page_content": str(row.to_dict()),
                    "metadata": {"source": path, "row": i, "type": "csv"}
                })

        elif ext == "json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data[:100]):
                    documents.append({
                        "page_content": str(item),
                        "metadata": {"source": path, "index": i, "type": "json"}
                    })
            else:
                documents.append({
                    "page_content": str(data)[:1000],
                    "metadata": {"source": path, "type": "json"}
                })

        elif ext == "parquet":
            df = pd.read_parquet(path)
            for i, row in df.head(50).iterrows():
                documents.append({
                    "page_content": str(row.to_dict()),
                    "metadata": {"source": path, "row": i, "type": "parquet"}
                })

        return documents

    def _chunk_text(self, text: str, chunk_size: int = 500,
                    overlap: int = 50) -> list:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def _load_pipeline_outputs(self) -> list:
        """Load VEDA pipeline output files as documents."""
        documents = []
        d = "outputs"

        json_files = [f for f in os.listdir(d) if f.endswith(".json")][:5]
        for filename in json_files:
            path = os.path.join(d, filename)
            try:
                with open(path) as f:
                    content = f.read()
                chunks = self._chunk_text(content, chunk_size=300, overlap=30)
                for i, chunk in enumerate(chunks[:3]):
                    documents.append({
                        "page_content": chunk,
                        "metadata": {
                            "source": filename,
                            "chunk": i,
                            "type": "veda_output"
                        }
                    })
            except:
                pass

        return documents

    def run(self, state: dict) -> dict:
        """
        Data Loader:
        1. Load dataset as documents
        2. Load pipeline outputs
        3. Chunk documents
        4. Save document store
        """

        dataset_path = state.get("dataset_path", "")
        documents = []

        if dataset_path and os.path.exists(dataset_path):
            self.log("Loading dataset: " + dataset_path)
            dataset_docs = self._load_document(dataset_path)
            documents.extend(dataset_docs)
            self.log("Loaded " + str(len(dataset_docs)) + " documents from dataset")

        self.log("Loading pipeline output files...")
        output_docs = self._load_pipeline_outputs()
        documents.extend(output_docs)
        self.log("Loaded " + str(len(output_docs)) + " documents from outputs")

        self.log("Total documents: " + str(len(documents)))

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        doc_store = {
            "total_documents": len(documents),
            "sources": list(set(d["metadata"]["source"] for d in documents)),
            "sample_documents": documents[:3]
        }

        path = "outputs/" + run_id + "_documents.json"
        with open(path, "w") as f:
            json.dump(doc_store, f, indent=2, default=str)

        state["lc_documents"] = doc_store
        state["raw_documents"] = documents
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] DataLoaderAgent: " +
            str(len(documents)) + " documents loaded"
        )

        self.log("=" * 50)
        self.log("DATA LOADER COMPLETE")
        self.log("Documents : " + str(len(documents)))
        self.log("Sources   : " + str(doc_store["sources"][:3]))
        self.log("=" * 50)

        return state