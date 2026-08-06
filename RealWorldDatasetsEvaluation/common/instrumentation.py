from __future__ import annotations
import builtins
import contextlib
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import psutil
except Exception:
    psutil = None


@dataclass
class Activity:
    detector_detections: int = 0
    adaptation_activations: int = 0
    sccm_adaptations: int = 0
    sccm_recalibrations: int = 0
    detection_indices: list[int] = field(default_factory=list)
    event_lines: list[str] = field(default_factory=list)


_DETECTION_PATTERNS = [
    re.compile(r"^(ADWIN|KSWIN) drift detected at mini-batch\s+(\d+)", re.I),
    re.compile(r"^(ADWIN|KSWIN) detected drift at global sample index:\s*(\d+)", re.I),
]


class EventPrintCollector:
    def __init__(self, activity: Activity, echo: bool = False):
        self.activity = activity
        self.echo = echo
        self.original = builtins.print

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        text = kwargs.get("sep", " ").join(str(x) for x in args)
        if self.echo:
            self.original(*args, **kwargs)
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            matched = False
            for pattern in _DETECTION_PATTERNS:
                match = pattern.search(line)
                if match:
                    self.activity.detector_detections += 1
                    self.activity.detection_indices.append(int(match.group(2)))
                    matched = True
                    break
            if line in {"RESET ACTIVATED", "WINDOW RETRAIN ACTIVATED", "SSPT ACTIVATED", "OHL ACTIVATED"}:
                self.activity.adaptation_activations += 1
                matched = True
            elif line.startswith("SSPT tuned ") or line.startswith("OHL tuned "):
                self.activity.adaptation_activations += 1
                matched = True
            if matched:
                self.activity.event_lines.append(line)


@contextlib.contextmanager
def capture_events(activity: Activity, echo: bool = False):
    collector = EventPrintCollector(activity, echo=echo)
    original = builtins.print
    builtins.print = collector
    try:
        yield
    finally:
        builtins.print = original


@contextlib.contextmanager
def count_sccm_calls(activity: Activity):
    from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
    original_st = ConceptDriftDetector.detect_ST_drift
    original_lt = ConceptDriftDetector.detect_LT_drift

    def wrapped_st(self, *args, **kwargs):
        result = original_st(self, *args, **kwargs)
        if bool(result):
            activity.sccm_adaptations += 1
        return result

    def wrapped_lt(self, *args, **kwargs):
        result = original_lt(self, *args, **kwargs)
        if bool(result):
            activity.sccm_recalibrations += 1
        return result

    ConceptDriftDetector.detect_ST_drift = wrapped_st
    ConceptDriftDetector.detect_LT_drift = wrapped_lt
    try:
        yield
    finally:
        ConceptDriftDetector.detect_ST_drift = original_st
        ConceptDriftDetector.detect_LT_drift = original_lt


@dataclass
class ResourceMeasurement:
    runtime_seconds: float
    rss_before_mb: float
    peak_rss_mb: float
    peak_rss_delta_mb: float
    measurement_method: str


class _Sampler:
    def __init__(self, interval=0.01):
        self.interval = max(0.001, float(interval))
        self.stop_event = threading.Event()
        self.process = psutil.Process(os.getpid()) if psutil else None
        self.before = self._rss()
        self.peak = self.before
        self.thread = None

    def _rss(self):
        if self.process is None:
            return 0
        try:
            return int(self.process.memory_info().rss)
        except Exception:
            return 0

    def _loop(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self._rss())
            self.stop_event.wait(self.interval)

    def start(self):
        if self.process:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        if self.process:
            self.peak = max(self.peak, self._rss())
            self.stop_event.set()
            if self.thread:
                self.thread.join(timeout=0.2)
            self.peak = max(self.peak, self._rss())


def measure(function: Callable[[], Any]):
    sampler = _Sampler()
    sampler.start()
    start = time.perf_counter()
    try:
        result = function()
    finally:
        elapsed = time.perf_counter() - start
        sampler.stop()
    mb = 1024.0 * 1024.0
    measurement = ResourceMeasurement(
        runtime_seconds=float(elapsed),
        rss_before_mb=float(sampler.before / mb),
        peak_rss_mb=float(sampler.peak / mb),
        peak_rss_delta_mb=float(max(0, sampler.peak - sampler.before) / mb),
        measurement_method="sampled_process_rss_psutil" if psutil else "unavailable",
    )
    return result, measurement


def per_1000(value: float, samples: int) -> float:
    return 0.0 if samples <= 0 else float(value) * 1000.0 / float(samples)
