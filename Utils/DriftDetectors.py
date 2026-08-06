"""Compatibility layer for the manuscript drift-detector implementation.

The manuscript specifies the scikit-multiflow reference detectors. This module
uses scikit-multiflow when available and exposes the small River-like API used
by the existing model code (``update`` + ``drift_detected``). A River fallback
is retained only to keep the project runnable on platforms where
scikit-multiflow cannot be built; the active backend is available through
``BACKEND`` and is written to validation reports.
"""
from __future__ import annotations

from typing import Any

BACKEND = "unavailable"
_SK_ADWIN: Any = None
_SK_KSWIN: Any = None
_RIVER_DRIFT: Any = None

try:  # Preferred, manuscript-aligned backend.
    from skmultiflow.drift_detection import ADWIN as _SK_ADWIN  # type: ignore
    from skmultiflow.drift_detection import KSWIN as _SK_KSWIN  # type: ignore
    BACKEND = "scikit-multiflow"
except Exception:
    try:  # Portable fallback.
        from river import drift as _RIVER_DRIFT  # type: ignore
        BACKEND = "river-fallback"
    except Exception:
        BACKEND = "unavailable"


def _require_backend() -> None:
    if BACKEND == "unavailable":
        raise ImportError(
            "No drift-detector backend is installed. Install dependencies from "
            "requirements.txt. scikit-multiflow is preferred; river is the "
            "documented compatibility fallback."
        )


class ADWIN:
    def __init__(self, delta: float = 0.002, **kwargs: Any) -> None:
        _require_backend()
        self._drift_detected = False
        if BACKEND == "scikit-multiflow":
            self._detector = _SK_ADWIN(delta=delta)
        else:
            self._detector = _RIVER_DRIFT.ADWIN(delta=delta, **kwargs)

    def update(self, value: float) -> "ADWIN":
        if BACKEND == "scikit-multiflow":
            self._detector.add_element(float(value))
            self._drift_detected = bool(self._detector.detected_change())
        else:
            self._detector.update(float(value))
            self._drift_detected = bool(self._detector.drift_detected)
        return self

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected


class KSWIN:
    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        **kwargs: Any,
    ) -> None:
        _require_backend()
        self._drift_detected = False
        if BACKEND == "scikit-multiflow":
            self._detector = _SK_KSWIN(
                alpha=alpha,
                window_size=window_size,
                stat_size=stat_size,
            )
        else:
            self._detector = _RIVER_DRIFT.KSWIN(
                alpha=alpha,
                window_size=window_size,
                stat_size=stat_size,
                **kwargs,
            )

    def update(self, value: float) -> "KSWIN":
        if BACKEND == "scikit-multiflow":
            self._detector.add_element(float(value))
            self._drift_detected = bool(self._detector.detected_change())
        else:
            self._detector.update(float(value))
            self._drift_detected = bool(self._detector.drift_detected)
        return self

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected
