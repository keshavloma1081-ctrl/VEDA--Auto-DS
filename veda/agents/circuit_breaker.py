import time, logging, traceback
from enum import Enum
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    recovery_timeout: int = 60
    success_threshold: int = 1
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    success_count: int = field(default=0, init=False)
    last_failure_time: Optional[float] = field(default=None, init=False)
    last_success_time: Optional[float] = field(default=None, init=False)
    total_calls: int = field(default=0, init=False)
    total_failures: int = field(default=0, init=False)
    total_successes: int = field(default=0, init=False)

    def _should_attempt(self):
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time >= self.recovery_timeout:
                log.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        return True

    def _on_success(self):
        self.total_successes += 1
        self.last_success_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                log.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def _on_failure(self, error):
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                log.error(f"Circuit {self.name}: CLOSED -> OPEN")
                self.state = CircuitState.OPEN

    def execute(self, func, *args, **kwargs):
        self.total_calls += 1
        if not self._should_attempt():
            retry_in = int(self.recovery_timeout - (time.time() - (self.last_failure_time or 0)))
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN. Retry in {retry_in}s")
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitOpenError:
            raise
        except Exception as e:
            self._on_failure(e)
            raise

    def get_status(self):
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_rate": round(self.total_failures / self.total_calls * 100, 2) if self.total_calls > 0 else 0,
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat() if self.last_failure_time else None,
        }

    def reset(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

class CircuitOpenError(Exception):
    pass

class CircuitBreakerRegistry:
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get(self, name, **kwargs):
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name=name, **kwargs)
        return self._breakers[name]

    def get_all_status(self):
        return [cb.get_status() for cb in self._breakers.values()]

    def reset_all(self):
        for cb in self._breakers.values():
            cb.reset()

    def get_open_circuits(self):
        return [n for n, cb in self._breakers.items() if cb.state == CircuitState.OPEN]

circuit_registry = CircuitBreakerRegistry()

@dataclass
class AgentResult:
    agent_name: str
    success: bool
    duration_seconds: float
    result: Optional[Any] = None
    error: Optional[str] = None
    fallback_used: bool = False
    skipped: bool = False
    retry_count: int = 0

    def to_dict(self):
        return {
            "agent": self.agent_name,
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 3),
            "fallback_used": self.fallback_used,
            "skipped": self.skipped,
            "retry_count": self.retry_count,
            "error": self.error,
        }

class PipelineFailureError(Exception):
    pass

class ResilientAgentExecutor:
    AGENT_FALLBACKS = {
        "DataIngestion":      "required",
        "DataCleaning":       "use_defaults",
        "DataProfiling":      "skip",
        "FeatureEngineering": "use_defaults",
        "FeatureSelection":   "skip",
        "FeatureScaling":     "use_defaults",
        "ModelTraining":      "required",
        "ModelEvaluation":    "use_defaults",
        "HyperparamTuning":   "skip",
        "ModelSelection":     "use_defaults",
        "ReportGeneration":   "skip",
    }

    def __init__(self, max_retries=2, base_delay=1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.execution_log: List[AgentResult] = []

    def execute_agent(self, agent_name, agent_func, state, *args, **kwargs):
        start_time = time.time()
        fallback_strategy = self.AGENT_FALLBACKS.get(agent_name, "skip")
        cb = circuit_registry.get(agent_name, failure_threshold=3, recovery_timeout=60)
        last_error = None
        retry_count = 0
        for attempt in range(self.max_retries + 1):
            try:
                result = cb.execute(agent_func, state, *args, **kwargs)
                duration = time.time() - start_time
                r = AgentResult(agent_name=agent_name, success=True, duration_seconds=duration, result=result, retry_count=retry_count)
                self.execution_log.append(r)
                return r
            except CircuitOpenError as e:
                duration = time.time() - start_time
                return self._apply_fallback(agent_name, str(e), fallback_strategy, duration, 0)
            except Exception as e:
                last_error = e
                retry_count = attempt
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), 30.0)
                    log.warning(f"Agent {agent_name} retry {attempt+1}: {e}")
                    time.sleep(delay)
                else:
                    duration = time.time() - start_time
                    return self._apply_fallback(agent_name, str(last_error), fallback_strategy, duration, retry_count)

    def _apply_fallback(self, agent_name, error, strategy, duration, retry_count):
        if strategy == "required":
            r = AgentResult(agent_name=agent_name, success=False, duration_seconds=duration, error=error, retry_count=retry_count)
            self.execution_log.append(r)
            raise PipelineFailureError(f"Required agent '{agent_name}' failed: {error}")
        elif strategy == "skip":
            r = AgentResult(agent_name=agent_name, success=False, duration_seconds=duration, error=error, skipped=True, retry_count=retry_count)
        else:
            r = AgentResult(agent_name=agent_name, success=False, duration_seconds=duration, error=error, fallback_used=True, retry_count=retry_count)
        self.execution_log.append(r)
        return r

    def get_execution_summary(self):
        return {
            "total_agents": len(self.execution_log),
            "successful": sum(1 for r in self.execution_log if r.success),
            "failed": sum(1 for r in self.execution_log if not r.success and not r.skipped),
            "skipped": sum(1 for r in self.execution_log if r.skipped),
            "fallbacks_used": sum(1 for r in self.execution_log if r.fallback_used),
            "total_duration_seconds": round(sum(r.duration_seconds for r in self.execution_log), 2),
            "agent_results": [r.to_dict() for r in self.execution_log],
            "circuit_breakers": circuit_registry.get_all_status(),
        }

def get_circuit_health():
    all_status = circuit_registry.get_all_status()
    open_circuits = circuit_registry.get_open_circuits()
    return {
        "total_circuits": len(all_status),
        "open_circuits": len(open_circuits),
        "open_circuit_names": open_circuits,
        "overall_health": "degraded" if open_circuits else "healthy",
        "circuits": all_status
    }
