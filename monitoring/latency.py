"""
Latency Monitoring for Production RAG
Tracks performance budgets and SLA compliance
"""
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import time
import statistics
import structlog

from backend.config import settings

logger = structlog.get_logger()


@dataclass
class LatencyStats:
    """Statistics for a latency metric"""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    samples: list = field(default_factory=list)
    budget_ms: Optional[float] = None
    violations: int = 0
    
    def record(self, latency_ms: float):
        """Record a latency measurement."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        
        # Keep last 1000 samples for percentile calculation
        self.samples.append(latency_ms)
        if len(self.samples) > 1000:
            self.samples.pop(0)
        
        # Check budget violation
        if self.budget_ms and latency_ms > self.budget_ms:
            self.violations += 1
    
    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0
    
    @property
    def p50_ms(self) -> float:
        return self._percentile(50)
    
    @property
    def p95_ms(self) -> float:
        return self._percentile(95)
    
    @property
    def p99_ms(self) -> float:
        return self._percentile(99)
    
    @property
    def sla_compliance(self) -> float:
        """Percentage of requests within budget."""
        if self.count == 0:
            return 100.0
        return ((self.count - self.violations) / self.count) * 100
    
    def _percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p / 100)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float('inf') else 0,
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "budget_ms": self.budget_ms,
            "sla_compliance_pct": round(self.sla_compliance, 2)
        }


class LatencyMonitor:
    """
    Production latency monitoring for RAG pipeline.
    Tracks each stage and overall end-to-end latency.
    """
    
    def __init__(self):
        self.metrics: Dict[str, LatencyStats] = {
            "retrieval": LatencyStats(budget_ms=settings.latency.retrieval_ms),
            "generation": LatencyStats(budget_ms=settings.latency.generation_ms),
            "total": LatencyStats(budget_ms=settings.latency.total_ms),
            "embedding": LatencyStats(budget_ms=100),
            "citation": LatencyStats(budget_ms=50),
            "confidence": LatencyStats(budget_ms=20)
        }
    
    @contextmanager
    def measure(self, metric_name: str):
        """Context manager for measuring latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.record(metric_name, latency_ms)
    
    def record(self, metric_name: str, latency_ms: float):
        """Record a latency measurement."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = LatencyStats()
        
        self.metrics[metric_name].record(latency_ms)
        
        # Log warning if approaching budget
        stats = self.metrics[metric_name]
        if stats.budget_ms:
            threshold = stats.budget_ms * settings.latency.warn_threshold_pct
            if latency_ms > threshold:
                logger.warning(
                    "latency_warning",
                    metric=metric_name,
                    latency_ms=round(latency_ms, 2),
                    budget_ms=stats.budget_ms,
                    threshold_pct=settings.latency.warn_threshold_pct
                )
    
    def get_stats(self, metric_name: str) -> Optional[Dict]:
        """Get statistics for a metric."""
        if metric_name in self.metrics:
            return self.metrics[metric_name].to_dict()
        return None
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get all latency statistics."""
        return {name: stats.to_dict() for name, stats in self.metrics.items()}
    
    def check_sla(self) -> Dict[str, bool]:
        """Check SLA compliance for all metrics."""
        compliance = {}
        for name, stats in self.metrics.items():
            if stats.budget_ms:
                compliance[name] = stats.sla_compliance >= 99.0  # 99% SLA target
        return compliance
    
    def reset(self):
        """Reset all metrics."""
        for stats in self.metrics.values():
            stats.count = 0
            stats.total_ms = 0.0
            stats.min_ms = float('inf')
            stats.max_ms = 0.0
            stats.samples = []
            stats.violations = 0


def timed(metric_name: str):
    """Decorator for measuring function latency."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with latency_monitor.measure(metric_name):
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with latency_monitor.measure(metric_name):
                return await func(*args, **kwargs)
        
        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def asyncio_iscoroutinefunction(func):
    """Check if function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


# Global monitor instance
latency_monitor = LatencyMonitor()

