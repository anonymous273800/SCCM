"""One-file SCCM alarm-quality experiment for the ORIGINAL code.

This version has all experiment settings fixed inside the file. No environment
variables are needed.

It assumes the original SCCM model files do NOT support `return_event_log=True`.
It does not modify any existing project file. It runs the original SCCM models,
captures/parses their console output, and builds the reviewer-requested
alarm-quality tables.

Place this file at:
    Experiments/003 AlarmQuality/one_file_sccm_alarm_quality_LONG_TERM_FIXED.py

Run:
    python "Experiments/003 AlarmQuality/one_file_sccm_alarm_quality_LONG_TERM_FIXED.py"

Fixed settings in this file:
    SELECTED_DATASET_NAMES = []      # empty means all 18 synthetic datasets
    SELECTED_SEEDS = DEFAULT_SEEDS   # usually Constants.SEEDS5
    CANDIDATE_SOURCE = "long_term"  # confirmed long-term SCCM alarms
    COOLDOWN_FACTOR = 1.0
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Experiments", "003 AlarmQuality", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENTS_CSV = os.path.join(OUTPUT_DIR, "sccm_alarm_events.csv")
BY_RUN_CSV = os.path.join(OUTPUT_DIR, "sccm_alarm_quality_by_run.csv")
FOR_PAPER_CSV = os.path.join(OUTPUT_DIR, "sccm_alarm_quality_for_paper.csv")
ERRORS_CSV = os.path.join(OUTPUT_DIR, "sccm_alarm_quality_errors.csv")
CONSOLE_LOG = os.path.join(OUTPUT_DIR, "sccm_alarm_quality_console.log")

# ---------------------------------------------------------------------
# Imports from the original project
# ---------------------------------------------------------------------

from Utils import Constants, Util  # noqa: E402
from Hyperparameters import Hyperparameter  # noqa: E402

from Datasets.Synthetic.Abrupt import ADS01, ADS02, ADS03, ADS04, ADS05, ADS06  # noqa: E402
from Datasets.Synthetic.Incremental import IDS01, IDS02, IDS03, IDS04, IDS05, IDS06  # noqa: E402
from Datasets.Synthetic.Gradual import GDS01, GDS02, GDS03, GDS04, GDS05, GDS06  # noqa: E402

from Models.OLR_WA import OLR_WA_SCCM  # noqa: E402
from Models.PA import PA_SCCM  # noqa: E402
from Models.RLS import RLS_SCCM  # noqa: E402
from Models.WidrowHoff import WidrowHoff_SCCM  # noqa: E402

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

TRAIN_PERCENT = 90
DEFAULT_SEEDS = getattr(Constants, "SEEDS5", [0, 1, 42, 123, 7])

DATASETS = [
    ("ADS01", "abrupt", ADS01.get_DS01),
    ("ADS02", "abrupt", ADS02.get_DS02),
    ("ADS03", "abrupt", ADS03.get_DS03),
    ("ADS04", "abrupt", ADS04.get_DS04),
    ("ADS05", "abrupt", ADS05.get_DS05),
    ("ADS06", "abrupt", ADS06.get_DS06),
    ("IDS01", "incremental", IDS01.get_IDS01),
    ("IDS02", "incremental", IDS02.get_IDS02),
    ("IDS03", "incremental", IDS03.get_IDS03),
    ("IDS04", "incremental", IDS04.get_IDS04),
    ("IDS05", "incremental", IDS05.get_IDS05),
    ("IDS06", "incremental", IDS06.get_IDS06),
    ("GDS01", "gradual", GDS01.get_GDS01),
    ("GDS02", "gradual", GDS02.get_GDS02),
    ("GDS03", "gradual", GDS03.get_GDS03),
    ("GDS04", "gradual", GDS04.get_GDS04),
    ("GDS05", "gradual", GDS05.get_GDS05),
    ("GDS06", "gradual", GDS06.get_GDS06),
]

# ---------------------------------------------------------------------
# Fixed settings, edit here only if you want a duplicate run type
# ---------------------------------------------------------------------

# Empty list means run all 18 synthetic datasets.
# Example for a smoke test: ["ADS01"]
SELECTED_DATASET_NAMES = []

# Use the same five-seed setting as the manuscript when Constants.SEEDS5 exists.
# Example for a smoke test: [42]
SELECTED_SEEDS = list(DEFAULT_SEEDS)

# Use long-term SCCM confirmation rather than raw short-term candidate triggers.
# Other possible values in this parser: "short_term", "recalibration", "adaptation_applied".
CANDIDATE_SOURCE = "long_term"

# Alarm episode consolidation window = COOLDOWN_FACTOR * tolerance.
# With 1.0, cooldown is 50 for 1k streams and 100 for 2k streams.
COOLDOWN_FACTOR = 1.0

# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def get_selected_datasets():
    if not SELECTED_DATASET_NAMES:
        return DATASETS
    selected_names = set(name.upper() for name in SELECTED_DATASET_NAMES)
    return [item for item in DATASETS if item[0].upper() in selected_names]


def get_selected_seeds():
    return [int(seed) for seed in SELECTED_SEEDS]


def safe_float(value):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def write_csv(rows, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

# ---------------------------------------------------------------------
# Dataset and drift-truth utilities
# ---------------------------------------------------------------------

def load_synthetic_dataset(dataset_function, seed):
    result = dataset_function(seed=seed, return_meta=True)
    X = result[0]
    y = result[1]
    meta = result[-1]
    return X, y, meta


def split_train_test(X, y):
    n_samples = X.shape[0]
    train_size = int(TRAIN_PERCENT * n_samples / 100)
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:], train_size


def get_known_drift_points(dataset_name, meta=None):
    dataset_name = dataset_name.upper()
    if meta:
        drift_type = str(meta.get("drift_type", "")).lower()
        if drift_type == "abrupt" and "drift_point" in meta:
            return [int(meta["drift_point"])]
        if drift_type == "incremental":
            samples_per_step = int(meta.get("samples_per_step", 0))
            n_steps = int(meta.get("n_steps", 0))
            if samples_per_step > 0 and n_steps > 1:
                return [samples_per_step * i for i in range(1, n_steps)]
        if drift_type == "gradual" and "segment_lengths" in meta:
            points = []
            running = 0
            for segment_length in list(meta["segment_lengths"])[:-1]:
                running += int(segment_length)
                points.append(running)
            return points

    if dataset_name.startswith("ADS"):
        ds_num = int(dataset_name[-2:])
        return [500] if ds_num in [1, 2, 3] else [1000]
    if dataset_name.startswith("IDS"):
        ds_num = int(dataset_name[-2:])
        return list(range(100, 1000, 100)) if ds_num in [1, 2, 3] else list(range(200, 2000, 200))
    if dataset_name.startswith("GDS"):
        ds_num = int(dataset_name[-2:])
        return [300, 400, 500, 600, 700] if ds_num in [1, 2, 3] else [600, 800, 1000, 1200, 1400]
    raise ValueError(f"Unknown synthetic dataset name: {dataset_name}")


def filter_drift_points_to_training(true_drift_points, train_length):
    return [int(p) for p in true_drift_points if int(p) < int(train_length)]

# ---------------------------------------------------------------------
# Console parsing for ORIGINAL SCCM code with no return_event_log flag
# ---------------------------------------------------------------------

class SCCMConsoleParser:
    """Parse SCCM printed output into event rows.

    This is intentionally lightweight. It uses the existing print statements:
      * *********** mini-batch- N *************
      * SHORT TERM DRIFT DETECTED True
      * Long Term Drift Detected True/False
      * inside while: additional mini-batch request # N
      * tuned_w_inc / tuned_C / tuned_lambda / tuned_learning_rate
    """

    MINI_BATCH_RE = re.compile(r"mini-batch-\s*(\d+)")
    ADDITIONAL_RE = re.compile(r"additional mini-batch request #\s*(\d+)")
    DRIFT_MAG_RE = re.compile(r"drift_magnitude\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    def __init__(self, model_name, increment_size, base_offset):
        self.model_name = model_name
        self.increment_size = int(increment_size)
        self.base_offset = int(base_offset)
        self.current_iteration = None
        self.events = []
        self.last_event = None
        self.pending_drift_magnitude = None

    def _batch_bounds(self, iteration):
        iteration = int(iteration)
        if self.model_name in {"PA-SCCM", "RLS-SCCM"}:
            # Original PA/RLS generator starts at j=0 and processes from index 0.
            batch_start = iteration * self.increment_size + 1
            batch_end = (iteration + 1) * self.increment_size
        else:
            # OLR-WA and Widrow-Hoff start after base-model training and j starts at 1.
            batch_start = self.base_offset + (iteration - 1) * self.increment_size + 1
            batch_end = self.base_offset + iteration * self.increment_size
        return batch_start, batch_end

    def parse_line(self, line):
        line = line.strip()
        if not line:
            return

        mini_match = self.MINI_BATCH_RE.search(line)
        if mini_match:
            self.current_iteration = int(mini_match.group(1))

        if "drift_magnitude" in line:
            matches = self.DRIFT_MAG_RE.findall(line)
            if matches:
                self.pending_drift_magnitude = safe_float(matches[-1])

        if "SHORT TERM DRIFT DETECTED" in line and "True" in line:
            if self.current_iteration is None:
                return
            batch_start, batch_end = self._batch_bounds(self.current_iteration)
            event = {
                "iteration": int(self.current_iteration),
                "batch_start": batch_start,
                "batch_end": batch_end,
                "alarm_index": batch_end,
                "short_term_drift_detected": True,
                "adaptation_triggered": True,
                "adaptation_applied": True,
                "long_term_drift_detected": False,
                "recalibration_triggered": False,
                "recalibration_batches": 0,
                "drift_magnitude": self.pending_drift_magnitude,
                "tuned_hyperparameter": "",
                "tuned_hyperparameter_value": None,
            }
            self.events.append(event)
            self.last_event = event
            return

        if self.last_event is not None and "Long Term Drift Detected" in line:
            is_long_term = "True" in line
            self.last_event["long_term_drift_detected"] = is_long_term
            if is_long_term:
                self.last_event["recalibration_triggered"] = True
            return

        if self.last_event is not None:
            additional_match = self.ADDITIONAL_RE.search(line)
            if additional_match:
                self.last_event["recalibration_triggered"] = True
                self.last_event["recalibration_batches"] = max(
                    int(self.last_event.get("recalibration_batches", 0) or 0),
                    int(additional_match.group(1)),
                )
                return

            if line.startswith("tuned_w_inc"):
                self.last_event["tuned_hyperparameter"] = "w_inc"
                self.last_event["tuned_hyperparameter_value"] = safe_float(line.split()[-1])
                return
            if line.startswith("tuned_C"):
                self.last_event["tuned_hyperparameter"] = "C"
                self.last_event["tuned_hyperparameter_value"] = safe_float(line.split()[-1])
                return
            if line.startswith("tuned_lambda"):
                self.last_event["tuned_hyperparameter"] = "lambda"
                self.last_event["tuned_hyperparameter_value"] = safe_float(line.split()[-1])
                return
            if line.startswith("tuned_learning_rate"):
                self.last_event["tuned_hyperparameter"] = "learning_rate"
                self.last_event["tuned_hyperparameter_value"] = safe_float(line.split()[-1])
                return


class ParsingTee(io.TextIOBase):
    def __init__(self, console_stream, log_stream, parser):
        self.console_stream = console_stream
        self.log_stream = log_stream
        self.parser = parser
        self.buffer = ""

    def write(self, data):
        self.console_stream.write(data)
        self.console_stream.flush()
        self.log_stream.write(data)
        self.log_stream.flush()
        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.parser.parse_line(line)
        return len(data)

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()
        if self.buffer:
            self.parser.parse_line(self.buffer)
            self.buffer = ""

# ---------------------------------------------------------------------
# Alarm consolidation and matching
# ---------------------------------------------------------------------

def select_candidate_events(event_log, source="short_term"):
    source = source.lower().strip()
    if source == "short_term":
        return [event for event in event_log if is_truthy(event.get("short_term_drift_detected"))]
    if source == "long_term":
        return [event for event in event_log if is_truthy(event.get("long_term_drift_detected"))]
    if source == "recalibration":
        return [event for event in event_log if is_truthy(event.get("recalibration_triggered"))]
    if source == "adaptation_applied":
        return [event for event in event_log if is_truthy(event.get("adaptation_applied"))]
    raise ValueError(f"Unknown candidate source: {source}")


def consolidate_alarm_episodes(candidate_events, cooldown):
    sorted_events = sorted(candidate_events, key=lambda event: int(event.get("alarm_index", 0)))
    episodes = []
    current = None
    current_end = -1
    for event in sorted_events:
        alarm_index = int(event.get("alarm_index", 0))
        if current is None or alarm_index > current_end:
            current = {
                "episode_alarm_index": alarm_index,
                "episode_start_index": alarm_index,
                "episode_end_index": alarm_index,
                "episode_size": 1,
                "has_long_term_drift": is_truthy(event.get("long_term_drift_detected")),
                "has_recalibration": is_truthy(event.get("recalibration_triggered")),
                "recalibration_batches": int(event.get("recalibration_batches", 0) or 0),
            }
            episodes.append(current)
            current_end = alarm_index + int(cooldown)
        else:
            current["episode_end_index"] = alarm_index
            current["episode_size"] += 1
            current["has_long_term_drift"] = bool(current["has_long_term_drift"] or is_truthy(event.get("long_term_drift_detected")))
            current["has_recalibration"] = bool(current["has_recalibration"] or is_truthy(event.get("recalibration_triggered")))
            current["recalibration_batches"] += int(event.get("recalibration_batches", 0) or 0)
    return episodes


def match_alarm_indices_to_drifts(alarm_indices, true_drift_points, tolerance):
    used_alarm_ids = set()
    delays = []
    matched_drift_points = []
    matched_alarm_indices = []
    tp = 0
    fn = 0
    for drift_point in true_drift_points:
        matched_alarm_id = None
        matched_delay = None
        for alarm_id, alarm_index in enumerate(alarm_indices):
            if alarm_id in used_alarm_ids:
                continue
            if int(drift_point) <= int(alarm_index) <= int(drift_point) + int(tolerance):
                matched_alarm_id = alarm_id
                matched_delay = int(alarm_index) - int(drift_point)
                break
        if matched_alarm_id is None:
            fn += 1
        else:
            tp += 1
            used_alarm_ids.add(matched_alarm_id)
            delays.append(int(matched_delay))
            matched_drift_points.append(int(drift_point))
            matched_alarm_indices.append(int(alarm_indices[matched_alarm_id]))
    fp = len(alarm_indices) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    delay_sum = sum(delays)
    delay_count = len(delays)
    mean_delay = delay_sum / delay_count if delay_count > 0 else None
    median_delay = None
    if delays:
        sorted_delays = sorted(delays)
        mid = len(sorted_delays) // 2
        median_delay = float(sorted_delays[mid]) if len(sorted_delays) % 2 else float(sorted_delays[mid - 1] + sorted_delays[mid]) / 2.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay": mean_delay,
        "median_delay": median_delay,
        "delay_sum": delay_sum,
        "delay_count": delay_count,
        "matched_drift_points": ";".join(map(str, matched_drift_points)),
        "matched_alarm_indices": ";".join(map(str, matched_alarm_indices)),
    }


def summarize_alarm_quality(event_log, true_drift_points, tolerance, candidate_source="short_term", cooldown=None):
    if cooldown is None:
        cooldown = tolerance
    candidate_events = select_candidate_events(event_log, source=candidate_source)
    episodes = consolidate_alarm_episodes(candidate_events, cooldown=cooldown)
    episode_indices = [int(ep["episode_alarm_index"]) for ep in episodes]
    matched = match_alarm_indices_to_drifts(episode_indices, true_drift_points, tolerance)
    raw_indices = sorted([int(event.get("alarm_index", 0)) for event in candidate_events])
    raw_matched = match_alarm_indices_to_drifts(raw_indices, true_drift_points, tolerance)
    candidate_triggers = len(candidate_events)
    alarm_episodes = len(episodes)
    adaptations = sum(1 for event in event_log if is_truthy(event.get("adaptation_triggered")))
    applied_adaptations = sum(1 for event in event_log if is_truthy(event.get("adaptation_applied")))
    recalibrations = sum(1 for event in event_log if is_truthy(event.get("recalibration_triggered")))
    recalibration_batches = sum(int(event.get("recalibration_batches", 0) or 0) for event in event_log)
    return {
        "true_drifts": len(true_drift_points),
        "candidate_triggers": candidate_triggers,
        "alarm_episodes": alarm_episodes,
        "confirmed_alarms": alarm_episodes,
        "duplicate_candidate_triggers": max(0, candidate_triggers - alarm_episodes),
        "candidate_source": candidate_source,
        "cooldown": int(cooldown),
        "tp": matched["tp"],
        "fp": matched["fp"],
        "fn": matched["fn"],
        "precision": matched["precision"],
        "recall": matched["recall"],
        "f1": matched["f1"],
        "mean_delay": matched["mean_delay"],
        "median_delay": matched["median_delay"],
        "delay_sum": matched["delay_sum"],
        "delay_count": matched["delay_count"],
        "matched_drift_points": matched["matched_drift_points"],
        "matched_alarm_indices": matched["matched_alarm_indices"],
        "raw_trigger_tp": raw_matched["tp"],
        "raw_trigger_fp": raw_matched["fp"],
        "raw_trigger_fn": raw_matched["fn"],
        "raw_trigger_precision": raw_matched["precision"],
        "raw_trigger_recall": raw_matched["recall"],
        "adaptations": adaptations,
        "applied_adaptations": applied_adaptations,
        "recalibrations": recalibrations,
        "recalibration_batches": recalibration_batches,
    }


def aggregate_for_paper(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["model"], row["drift_type"])].append(row)
    paper_rows = []
    for (model, drift_type), rows in grouped.items():
        true_drifts = sum(int(r["true_drifts"]) for r in rows)
        candidate_triggers = sum(int(r["candidate_triggers"]) for r in rows)
        alarm_episodes = sum(int(r["alarm_episodes"]) for r in rows)
        duplicate_candidate_triggers = sum(int(r["duplicate_candidate_triggers"]) for r in rows)
        tp = sum(int(r["tp"]) for r in rows)
        fp = sum(int(r["fp"]) for r in rows)
        fn = sum(int(r["fn"]) for r in rows)
        delay_sum = sum(float(r.get("delay_sum", 0) or 0) for r in rows)
        delay_count = sum(int(r.get("delay_count", 0) or 0) for r in rows)
        adaptations = sum(int(r.get("adaptations", 0) or 0) for r in rows)
        applied_adaptations = sum(int(r.get("applied_adaptations", 0) or 0) for r in rows)
        recalibrations = sum(int(r.get("recalibrations", 0) or 0) for r in rows)
        recalibration_batches = sum(int(r.get("recalibration_batches", 0) or 0) for r in rows)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_delay = delay_sum / delay_count if delay_count > 0 else None
        median_values = [float(r["median_delay"]) for r in rows if r.get("median_delay") not in [None, ""]]
        median_delay = sum(median_values) / len(median_values) if median_values else None
        raw_tp = sum(int(r.get("raw_trigger_tp", 0) or 0) for r in rows)
        raw_fp = sum(int(r.get("raw_trigger_fp", 0) or 0) for r in rows)
        raw_fn = sum(int(r.get("raw_trigger_fn", 0) or 0) for r in rows)
        raw_precision = raw_tp / (raw_tp + raw_fp) if (raw_tp + raw_fp) > 0 else 0.0
        raw_recall = raw_tp / (raw_tp + raw_fn) if (raw_tp + raw_fn) > 0 else 0.0
        paper_rows.append({
            "model": model,
            "drift_type": drift_type,
            "true_drifts": true_drifts,
            "candidate_triggers": candidate_triggers,
            "alarm_episodes": alarm_episodes,
            "confirmed_alarms": alarm_episodes,
            "duplicate_candidate_triggers": duplicate_candidate_triggers,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_delay": round(mean_delay, 2) if mean_delay is not None else "",
            "median_delay": round(median_delay, 2) if median_delay is not None else "",
            "raw_trigger_precision": round(raw_precision, 4),
            "raw_trigger_recall": round(raw_recall, 4),
            "adaptations": adaptations,
            "applied_adaptations": applied_adaptations,
            "recalibrations": recalibrations,
            "recalibration_batches": recalibration_batches,
        })
    model_order = {"OLR-WA-SCCM": 0, "PA-SCCM": 1, "RLS-SCCM": 2, "WidrowHoff-SCCM": 3}
    drift_order = {"abrupt": 0, "incremental": 1, "gradual": 2}
    paper_rows.sort(key=lambda r: (model_order.get(r["model"], 99), drift_order.get(r["drift_type"], 99)))
    return paper_rows

# ---------------------------------------------------------------------
# Original model runners
# ---------------------------------------------------------------------

def run_olr_wa_original(X_train, y_train, X_test, y_test, dataset_name):
    n_features = X_train.shape[1]
    increment_size = Hyperparameter.olr_wa_increment_size(n_features, user_defined_val=10)
    n_samples_tot = X_train.shape[0] + X_test.shape[0]
    base_offset = Util.calculate_no_of_base_model_points(n_samples_tot, Hyperparameter.olr_wa_base_model_size0)
    OLR_WA_SCCM.olr_wa_sccm(
        X_train, y_train,
        Hyperparameter.olr_wa_w_base,
        Hyperparameter.olr_wa_w_inc,
        Hyperparameter.olr_wa_base_model_size0,
        increment_size,
        X_test, y_test,
        kpi="R2",
        multiplier=2.5,
    )
    return increment_size, base_offset


def run_pa_original(X_train, y_train, X_test, y_test, dataset_name):
    increment_size = 1
    base_offset = 0
    PA_SCCM.ad_pa_generic(
        X_train, y_train,
        c=1.0,
        epsilon=0.1,
        X_test=X_test,
        y_test=y_test,
        kpi="MSE",
        multiplier=1.5,
        report_interval=1,
        ds=dataset_name,
        c_bounds=(0.05, 10.0),
    )
    return increment_size, base_offset


def run_rls_original(X_train, y_train, X_test, y_test, dataset_name):
    increment_size = 1
    base_offset = 0
    RLS_SCCM.ad_rls_generic(
        X_train, y_train,
        lambda_=0.99,
        delta=1.0,
        X_test=X_test,
        y_test=y_test,
        kpi="MSE",
        multiplier=1.5,
        DS=dataset_name,
        report_interval=1,
        lambda_bounds=(0.85, 0.999),
    )
    return increment_size, base_offset


def run_wh_original(X_train, y_train, X_test, y_test, dataset_name):
    increment_size = 1
    n_samples_tot = X_train.shape[0] + X_test.shape[0]
    base_offset = Util.calculate_no_of_base_model_points(n_samples_tot, 1)
    WidrowHoff_SCCM.ad_widrow_hoff_generic(
        X_train, y_train,
        learning_rate=0.01,
        X_test=X_test,
        y_test=y_test,
        kpi="MSE",
        multiplier=1.5,
        DS=dataset_name,
        report_interval=1,
    )
    return increment_size, base_offset


MODEL_RUNNERS = [
    ("OLR-WA-SCCM", run_olr_wa_original),
    ("PA-SCCM", run_pa_original),
    ("RLS-SCCM", run_rls_original),
    ("WidrowHoff-SCCM", run_wh_original),
]

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_model_and_parse(model_name, runner, X_train, y_train, X_test, y_test, dataset_name, log_file):
    # Create parser with a temporary increment/base, then update before parsing if possible.
    # Because output parsing needs increment/base during execution, we compute those same values here.
    if model_name == "OLR-WA-SCCM":
        n_features = X_train.shape[1]
        increment_size = Hyperparameter.olr_wa_increment_size(n_features, user_defined_val=10)
        base_offset = Util.calculate_no_of_base_model_points(X_train.shape[0] + X_test.shape[0], Hyperparameter.olr_wa_base_model_size0)
    elif model_name in {"PA-SCCM", "RLS-SCCM"}:
        increment_size = 1
        base_offset = 0
    else:
        increment_size = 1
        base_offset = Util.calculate_no_of_base_model_points(X_train.shape[0] + X_test.shape[0], 1)

    parser = SCCMConsoleParser(model_name=model_name, increment_size=increment_size, base_offset=base_offset)
    tee = ParsingTee(sys.stdout, log_file, parser)
    with redirect_stdout(tee), redirect_stderr(tee):
        runner(X_train, y_train, X_test, y_test, dataset_name)
    tee.flush()
    return parser.events


def attach_metadata_to_events(event_log, model_name, dataset_name, drift_type, seed):
    enriched = []
    for event in event_log:
        row = dict(event)
        row["model"] = model_name
        row["dataset"] = dataset_name
        row["drift_type"] = drift_type
        row["seed"] = seed
        enriched.append(row)
    return enriched


def run_all():
    warnings.filterwarnings("ignore")
    selected_datasets = get_selected_datasets()
    selected_seeds = get_selected_seeds()
    candidate_source = CANDIDATE_SOURCE
    cooldown_factor = float(COOLDOWN_FACTOR)

    all_event_rows = []
    all_run_summary_rows = []
    error_rows = []

    with open(CONSOLE_LOG, "w", encoding="utf-8") as log_file:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=log_file)
            log_file.flush()

        log_print("SCCM one-file alarm-quality experiment for ORIGINAL code, fixed long-term settings")
        log_print("Started:", datetime.now().isoformat(timespec="seconds"))
        log_print("Project root:", PROJECT_ROOT)
        log_print("Output dir:", OUTPUT_DIR)
        log_print("Datasets:", [name for name, _, _ in selected_datasets])
        log_print("Seeds:", selected_seeds)
        log_print("Candidate source:", candidate_source)
        log_print("Cooldown factor:", cooldown_factor)
        log_print("=" * 80)

        for dataset_name, drift_type, dataset_function in selected_datasets:
            for seed in selected_seeds:
                log_print(f"\nRunning dataset={dataset_name}, drift_type={drift_type}, seed={seed}")
                try:
                    X, y, meta = load_synthetic_dataset(dataset_function, seed)
                    X_train, y_train, X_test, y_test, train_length = split_train_test(X, y)
                    stream_length = X.shape[0]
                    true_drift_points = get_known_drift_points(dataset_name, meta=meta)
                    true_drift_points = filter_drift_points_to_training(true_drift_points, train_length)
                    tolerance = int(round(0.05 * stream_length))
                    cooldown = int(round(cooldown_factor * tolerance))
                    log_print("True drift points:", true_drift_points)
                    log_print("Tolerance:", tolerance)
                    log_print("Cooldown:", cooldown)

                    for model_name, runner in MODEL_RUNNERS:
                        log_print(f"\n  Model: {model_name}")
                        try:
                            event_log = run_model_and_parse(model_name, runner, X_train, y_train, X_test, y_test, dataset_name, log_file)
                            event_rows = attach_metadata_to_events(event_log, model_name, dataset_name, drift_type, seed)
                            all_event_rows.extend(event_rows)
                            summary = summarize_alarm_quality(
                                event_log=event_log,
                                true_drift_points=true_drift_points,
                                tolerance=tolerance,
                                candidate_source=candidate_source,
                                cooldown=cooldown,
                            )
                            summary["model"] = model_name
                            summary["dataset"] = dataset_name
                            summary["drift_type"] = drift_type
                            summary["seed"] = seed
                            summary["train_length"] = train_length
                            summary["stream_length"] = stream_length
                            summary["tolerance"] = tolerance
                            all_run_summary_rows.append(summary)
                            log_print(
                                "    Summary:",
                                f"candidate_triggers={summary['candidate_triggers']},",
                                f"alarm_episodes={summary['alarm_episodes']},",
                                f"TP={summary['tp']},",
                                f"FP={summary['fp']},",
                                f"FN={summary['fn']},",
                                f"precision={summary['precision']:.4f},",
                                f"recall={summary['recall']:.4f},",
                                f"F1={summary['f1']:.4f},",
                                f"mean_delay={summary['mean_delay']}",
                            )
                        except Exception as model_error:
                            tb = traceback.format_exc()
                            error_row = {"dataset": dataset_name, "drift_type": drift_type, "seed": seed, "model": model_name, "error": repr(model_error), "traceback": tb}
                            error_rows.append(error_row)
                            log_print("    ERROR:", repr(model_error))
                            log_print(tb)
                except Exception as dataset_error:
                    tb = traceback.format_exc()
                    error_row = {"dataset": dataset_name, "drift_type": drift_type, "seed": seed, "model": "DATASET_LOAD", "error": repr(dataset_error), "traceback": tb}
                    error_rows.append(error_row)
                    log_print("DATASET ERROR:", repr(dataset_error))
                    log_print(tb)

        paper_rows = aggregate_for_paper(all_run_summary_rows)
        write_csv(all_event_rows, EVENTS_CSV)
        write_csv(all_run_summary_rows, BY_RUN_CSV)
        write_csv(paper_rows, FOR_PAPER_CSV)
        write_csv(error_rows, ERRORS_CSV)

        log_print("\n" + "=" * 80)
        log_print("Saved:")
        log_print(" -", EVENTS_CSV)
        log_print(" -", BY_RUN_CSV)
        log_print(" -", FOR_PAPER_CSV)
        log_print(" -", ERRORS_CSV)
        log_print(" -", CONSOLE_LOG)
        log_print("Finished:", datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    run_all()
