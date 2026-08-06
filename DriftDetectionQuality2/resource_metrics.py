from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    import psutil
except Exception:  # pragma: no cover - fallback for minimal environments
    psutil = None


@dataclass
class ResourceMeasurement:
    runtime_seconds: float
    rss_before_mb: float
    peak_rss_mb: float
    peak_rss_delta_mb: float
    measurement_method: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "runtime_seconds": self.runtime_seconds,
            "rss_before_mb": self.rss_before_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "peak_rss_delta_mb": self.peak_rss_delta_mb,
            "memory_measurement_method": self.measurement_method,
        }


class _PeakRssSampler:
    def __init__(self, interval_seconds: float = 0.01):
        self.interval_seconds = max(0.001, float(interval_seconds))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self.before_bytes = self._rss_bytes()
        self.peak_bytes = self.before_bytes

    def _rss_bytes(self) -> int:
        if self._process is None:
            return 0
        try:
            return int(self._process.memory_info().rss)
        except Exception:
            return 0

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self.peak_bytes = max(self.peak_bytes, self._rss_bytes())
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self._process is None:
            return
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._process is None:
            return
        self.peak_bytes = max(self.peak_bytes, self._rss_bytes())
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.interval_seconds * 4))
        self.peak_bytes = max(self.peak_bytes, self._rss_bytes())


def measure_call(
    function: Callable[..., Any],
    *args: Any,
    sample_interval_seconds: float = 0.01,
    **kwargs: Any,
) -> tuple[Any, ResourceMeasurement]:
    """Execute one method and measure wall time plus sampled process RSS.

    RSS sampling captures Python and native allocations made by NumPy/SciPy. The
    reported delta is the maximum sampled RSS above the process RSS immediately
    before the method call. It is intentionally measured per method run.
    """

    sampler = _PeakRssSampler(sample_interval_seconds)
    sampler.start()
    start = time.perf_counter()
    try:
        result = function(*args, **kwargs)
    finally:
        runtime_seconds = time.perf_counter() - start
        sampler.stop()

    bytes_to_mb = 1024.0 * 1024.0
    before_mb = sampler.before_bytes / bytes_to_mb
    peak_mb = sampler.peak_bytes / bytes_to_mb
    measurement = ResourceMeasurement(
        runtime_seconds=runtime_seconds,
        rss_before_mb=before_mb,
        peak_rss_mb=peak_mb,
        peak_rss_delta_mb=max(0.0, peak_mb - before_mb),
        measurement_method=(
            "sampled_process_rss_psutil"
            if psutil is not None
            else "unavailable"
        ),
    )
    return result, measurement


def per_1000(count: int | float, processed_samples: int | float) -> float:
    try:
        denominator = float(processed_samples)
        if denominator <= 0:
            return 0.0
        return float(count) * 1000.0 / denominator
    except Exception:
        return 0.0
