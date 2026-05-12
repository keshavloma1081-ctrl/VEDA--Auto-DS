"""
VEDA — Autonomous Data Science System
agents/streaming/stream_ingest.py — Stream Ingest Agent

Simulates real-time data ingestion:
- Kafka-style message simulation
- Event generation
- Schema validation
- Backpressure handling
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

from veda.core.base_agent import BaseAgent


class StreamIngestAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="StreamIngestAgent",
            domain="streaming",
            version="1.0.0"
        )
        self.buffer = deque(maxlen=1000)
        self.total_messages = 0
        self.failed_messages = 0

    def _generate_stream_event(self, event_id: int,
                                schema: dict) -> dict:
        """Generate a single stream event."""
        np.random.seed(event_id)
        event = {
            "event_id": event_id,
            "timestamp": (datetime.now() - timedelta(
                seconds=np.random.randint(0, 3600)
            )).isoformat(),
            "source": np.random.choice(["web", "mobile", "api", "batch"]),
        }
        for field, dtype in schema.items():
            if dtype == "float":
                event[field] = round(float(np.random.normal(50, 15)), 4)
            elif dtype == "int":
                event[field] = int(np.random.randint(0, 100))
            elif dtype == "category":
                event[field] = np.random.choice(["A", "B", "C", "D"])
            elif dtype == "bool":
                event[field] = bool(np.random.choice([True, False]))
        return event

    def _validate_event(self, event: dict, schema: dict) -> bool:
        """Validate event against schema."""
        for field in schema:
            if field not in event:
                return False
            if event[field] is None:
                return False
        return True

    def _simulate_kafka_ingestion(self, n_events: int = 500,
                                   events_per_second: int = 100) -> dict:
        """Simulate Kafka-style message ingestion."""
        schema = {
            "value": "float",
            "quantity": "int",
            "category": "category",
            "is_fraud": "bool",
            "latency_ms": "float",
            "user_score": "float"
        }

        events = []
        failed = 0
        start_time = time.perf_counter()

        for i in range(n_events):
            event = self._generate_stream_event(i, schema)

            # Simulate occasional failures
            if np.random.random() < 0.02:
                failed += 1
                continue

            if self._validate_event(event, schema):
                self.buffer.append(event)
                events.append(event)
            else:
                failed += 1

        elapsed = time.perf_counter() - start_time
        throughput = round(len(events) / max(elapsed, 0.001), 2)

        self.total_messages = len(events)
        self.failed_messages = failed

        return {
            "total_generated": n_events,
            "successfully_ingested": len(events),
            "failed": failed,
            "failure_rate": round(failed / n_events * 100, 2),
            "throughput_per_sec": throughput,
            "elapsed_sec": round(elapsed, 4),
            "buffer_size": len(self.buffer),
            "schema": schema,
            "sample_events": events[:3]
        }

    def _compute_lag_metrics(self, events: list) -> dict:
        """Compute consumer lag metrics."""
        if not events:
            return {}
        timestamps = [datetime.fromisoformat(e["timestamp"]) for e in events[:100]]
        now = datetime.now()
        lags = [(now - ts).total_seconds() for ts in timestamps]
        return {
            "avg_lag_sec": round(float(np.mean(lags)), 2),
            "max_lag_sec": round(float(np.max(lags)), 2),
            "min_lag_sec": round(float(np.min(lags)), 2),
            "p95_lag_sec": round(float(np.percentile(lags, 95)), 2)
        }

    def run(self, state: dict) -> dict:
        self.log("Starting stream ingestion simulation...")
        self.log("Simulating Kafka producer -> consumer pipeline")

        ingest_results = self._simulate_kafka_ingestion(n_events=500)

        self.log("Ingested: " + str(ingest_results["successfully_ingested"]) +
                "/" + str(ingest_results["total_generated"]) + " events")
        self.log("Throughput: " + str(ingest_results["throughput_per_sec"]) + " events/sec")
        self.log("Failure rate: " + str(ingest_results["failure_rate"]) + "%")

        lag_metrics = self._compute_lag_metrics(
            list(self.buffer)
        )
        self.log("Avg lag: " + str(lag_metrics.get("avg_lag_sec", 0)) + "s")

        stream_data = pd.DataFrame(list(self.buffer))

        ingest_results["lag_metrics"] = lag_metrics

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        stream_path = "outputs/" + run_id + "_stream_data.parquet"
        stream_data.to_parquet(stream_path, index=False)

        results_path = "outputs/" + run_id + "_ingest_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {k: v for k, v in ingest_results.items() if k != "sample_events"},
                f, indent=2
            )

        state["stream_ingest"] = ingest_results
        state["stream_data"] = stream_data
        state["stream_buffer"] = list(self.buffer)
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] StreamIngestAgent: " +
            str(ingest_results["successfully_ingested"]) + " events, " +
            str(ingest_results["throughput_per_sec"]) + " eps"
        )

        self.log("=" * 50)
        self.log("STREAM INGEST COMPLETE")
        self.log("Events    : " + str(ingest_results["successfully_ingested"]))
        self.log("Throughput: " + str(ingest_results["throughput_per_sec"]) + " eps")
        self.log("Lag (avg) : " + str(lag_metrics.get("avg_lag_sec", 0)) + "s")
        self.log("=" * 50)

        return state