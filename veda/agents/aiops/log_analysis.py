"""
VEDA — Autonomous Data Science System
agents/aiops/log_analysis.py — Log Analysis Agent

Parses and analyzes system logs:
- Log level distribution
- Error pattern detection
- Frequency analysis
- Timeline analysis
- LLM-powered log summarization
"""

import os
import re
import json
import random
from datetime import datetime, timedelta
from collections import Counter
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class LogAnalysisAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="LogAnalysisAgent",
            domain="aiops",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

        self.log_pattern = re.compile(
            r"(?P<timestamp>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})"
            r".*?(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)"
            r".*?(?P<message>.+)"
        )

        self.error_patterns = {
            "OutOfMemory": r"out.of.memory|OOM|memory.exceeded",
            "NullPointer": r"null.pointer|NullPointer|NoneType",
            "ConnectionError": r"connection.refused|connection.timeout|ECONNREFUSED",
            "DiskFull": r"disk.full|no.space.left|ENOSPC",
            "PermissionDenied": r"permission.denied|EACCES|unauthorized",
            "Timeout": r"timeout|timed.out|deadline.exceeded",
            "DatabaseError": r"database.error|SQL.error|connection.pool",
            "ModelError": r"model.failed|prediction.error|inference.error"
        }

    def _generate_synthetic_logs(self, n_lines: int = 500) -> list:
        """Generate synthetic system logs for testing."""
        log_templates = {
            "INFO": [
                "Pipeline started successfully",
                "Model loaded from checkpoint",
                "Data ingestion complete: {} rows processed",
                "Feature engineering complete: {} features",
                "API request received from {}",
                "Prediction completed in {}ms",
                "Health check passed",
                "Cache hit for key: {}",
            ],
            "WARNING": [
                "High memory usage: {}%",
                "Slow query detected: {}ms",
                "Retry attempt {} of 3",
                "Cache miss for key: {}",
                "Rate limit approaching: {}/1000 requests",
                "Model drift detected: score={}",
            ],
            "ERROR": [
                "Connection refused to database",
                "Model prediction failed: NullPointerException",
                "Out of memory error during batch processing",
                "Timeout after 30s waiting for response",
                "Permission denied accessing /data/models",
                "Disk full: no space left on device",
            ],
            "CRITICAL": [
                "System overload: CPU usage 99%",
                "Database connection pool exhausted",
                "Model serving endpoint down",
            ]
        }

        logs = []
        base_time = datetime.now() - timedelta(hours=24)

        for i in range(n_lines):
            timestamp = base_time + timedelta(seconds=i * 170)

            rand = random.random()
            if rand < 0.65:
                level = "INFO"
            elif rand < 0.85:
                level = "WARNING"
            elif rand < 0.97:
                level = "ERROR"
            else:
                level = "CRITICAL"

            template = random.choice(log_templates[level])
            if "{}" in template:
                message = template.format(random.randint(10, 1000))
            else:
                message = template

            log_line = (timestamp.strftime("%Y-%m-%d %H:%M:%S") +
                       " [" + level + "] " + message)
            logs.append(log_line)

        return logs

    def _parse_logs(self, log_lines: list) -> list:
        """Parse log lines into structured records."""
        parsed = []
        for line in log_lines:
            match = self.log_pattern.search(line)
            if match:
                parsed.append({
                    "timestamp": match.group("timestamp"),
                    "level": match.group("level"),
                    "message": match.group("message").strip()
                })
            else:
                level = "INFO"
                for l in ["CRITICAL", "ERROR", "WARNING", "WARN", "DEBUG"]:
                    if l in line.upper():
                        level = l
                        break
                parsed.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": level,
                    "message": line.strip()
                })
        return parsed

    def _analyze_levels(self, parsed_logs: list) -> dict:
        """Analyze log level distribution."""
        levels = [log["level"] for log in parsed_logs]
        counter = Counter(levels)
        total = len(levels)
        return {
            level: {
                "count": count,
                "percentage": round(count / total * 100, 2)
            }
            for level, count in counter.most_common()
        }

    def _detect_error_patterns(self, parsed_logs: list) -> dict:
        """Detect known error patterns in logs."""
        error_logs = [log for log in parsed_logs if log["level"] in ["ERROR", "CRITICAL"]]
        detected = {}

        for pattern_name, pattern in self.error_patterns.items():
            matches = [
                log for log in error_logs
                if re.search(pattern, log["message"], re.IGNORECASE)
            ]
            if matches:
                detected[pattern_name] = {
                    "count": len(matches),
                    "sample": matches[0]["message"][:100]
                }

        return detected

    def _compute_error_rate(self, parsed_logs: list) -> dict:
        """Compute error rate over time."""
        total = len(parsed_logs)
        errors = sum(1 for log in parsed_logs if log["level"] in ["ERROR", "CRITICAL"])
        warnings = sum(1 for log in parsed_logs if log["level"] in ["WARNING", "WARN"])

        return {
            "total_logs": total,
            "error_count": errors,
            "warning_count": warnings,
            "error_rate": round(errors / total * 100, 2) if total > 0 else 0,
            "warning_rate": round(warnings / total * 100, 2) if total > 0 else 0,
            "health_score": round(max(0, 100 - (errors * 2) - (warnings * 0.5)), 2)
        }

    def _generate_summary(self, analysis: dict) -> str:
        """Generate LLM summary of log analysis."""
        prompt = """Analyze these system log statistics and provide a brief assessment.

Log Statistics:
""" + json.dumps(analysis, indent=2)[:1500] + """

Write 3 sentences covering:
1. Overall system health
2. Most critical issues found
3. Recommended immediate actions

Be specific and actionable."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "Log analysis complete. Review error patterns for action items."

    def run(self, state: dict) -> dict:
        """
        Log Analysis:
        1. Load or generate logs
        2. Parse log lines
        3. Analyze level distribution
        4. Detect error patterns
        5. Compute error rate
        6. Generate summary
        """

        self.log("Generating synthetic system logs...")
        log_lines = self._generate_synthetic_logs(n_lines=500)
        self.log("Generated " + str(len(log_lines)) + " log lines")

        self.log("Parsing logs...")
        parsed_logs = self._parse_logs(log_lines)

        self.log("Analyzing log levels...")
        level_dist = self._analyze_levels(parsed_logs)

        self.log("Detecting error patterns...")
        error_patterns = self._detect_error_patterns(parsed_logs)

        self.log("Computing error rates...")
        error_rate = self._compute_error_rate(parsed_logs)

        self.log("Generating summary...")
        analysis = {
            "level_distribution": level_dist,
            "error_patterns": error_patterns,
            "error_rate": error_rate
        }
        summary = self._generate_summary(analysis)

        for level, stats in level_dist.items():
            self.log(level + ": " + str(stats["count"]) +
                    " (" + str(stats["percentage"]) + "%)")

        if error_patterns:
            self.log("Error patterns detected: " + str(list(error_patterns.keys())))

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        log_analysis = {
            "total_logs": len(parsed_logs),
            "level_distribution": level_dist,
            "error_patterns": error_patterns,
            "error_rate": error_rate,
            "summary": summary
        }

        path = "outputs/" + run_id + "_log_analysis.json"
        with open(path, "w") as f:
            json.dump(log_analysis, f, indent=2)

        state["log_analysis"] = log_analysis
        state["parsed_logs"] = parsed_logs
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] LogAnalysisAgent: " +
            str(len(parsed_logs)) + " logs, error_rate=" +
            str(error_rate["error_rate"]) + "%"
        )

        self.log("=" * 50)
        self.log("LOG ANALYSIS COMPLETE")
        self.log("Total logs   : " + str(len(parsed_logs)))
        self.log("Error rate   : " + str(error_rate["error_rate"]) + "%")
        self.log("Health score : " + str(error_rate["health_score"]))
        self.log("Summary      : " + summary[:150] + "...")
        self.log("=" * 50)

        return state