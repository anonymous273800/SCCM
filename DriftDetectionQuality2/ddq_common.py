from __future__ import annotations

import csv
import importlib
import io
import math
import os
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path


def find_project_root(start_path: str | os.PathLike) -> Path:
    path = Path(start_path).resolve()

    if path.is_file():
        path = path.parent

    while True:
        if (
            (path / "Models").is_dir()
            and (path / "Datasets").is_dir()
            and (path / "Utils").is_dir()
        ):
            return path

        if path.parent == path:
            raise RuntimeError("Could not find SCCM project root.")

        path = path.parent


PROJECT_ROOT = find_project_root(__file__)
DDQ_ROOT = PROJECT_ROOT / "DriftDetectionQuality2"

for path in (str(PROJECT_ROOT), str(DDQ_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from Hyperparameters import Hyperparameter
from Utils import Util
from Models.OLR_WA import (
    OLR_WA_SCCM_DriftQuality as OLR_WA_SCCM,
)
from Models.PA import PA_SCCM_DriftQuality as PA_SCCM
from Models.RLS import RLS_SCCM_DriftQuality as RLS_SCCM
from Models.WidrowHoff import (
    WidrowHoff_SCCM_DriftQuality as WidrowHoff_SCCM,
)
from resource_metrics import measure_call, per_1000


DEFAULT_SEEDS = [0, 1, 42, 123, 7]
DEFAULT_SCCM_WINDOW_SIZE = 4
DEFAULT_USED_KPI_WINDOW_SIZE = 4


DATASET_REGISTRY = {
    "ADS01": (
        "Datasets.Synthetic2.Abrupt.ADS01",
        "get_DS01",
        "abrupt",
    ),
    "ADS02": (
        "Datasets.Synthetic2.Abrupt.ADS02",
        "get_DS02",
        "abrupt",
    ),
    "ADS03": (
        "Datasets.Synthetic2.Abrupt.ADS03",
        "get_DS03",
        "abrupt",
    ),
    "ADS04": (
        "Datasets.Synthetic2.Abrupt.ADS04",
        "get_DS04",
        "abrupt",
    ),
    "ADS05": (
        "Datasets.Synthetic2.Abrupt.ADS05",
        "get_DS05",
        "abrupt",
    ),
    "ADS06": (
        "Datasets.Synthetic2.Abrupt.ADS06",
        "get_DS06",
        "abrupt",
    ),
    "IDS01": (
        "Datasets.Synthetic2.Incremental.IDS01",
        "get_IDS01",
        "incremental",
    ),
    "IDS02": (
        "Datasets.Synthetic2.Incremental.IDS02",
        "get_IDS02",
        "incremental",
    ),
    "IDS03": (
        "Datasets.Synthetic2.Incremental.IDS03",
        "get_IDS03",
        "incremental",
    ),
    "IDS04": (
        "Datasets.Synthetic2.Incremental.IDS04",
        "get_IDS04",
        "incremental",
    ),
    "IDS05": (
        "Datasets.Synthetic2.Incremental.IDS05",
        "get_IDS05",
        "incremental",
    ),
    "IDS06": (
        "Datasets.Synthetic2.Incremental.IDS06",
        "get_IDS06",
        "incremental",
    ),
    "GDS01": (
        "Datasets.Synthetic2.Gradual.GDS01",
        "get_GDS01",
        "gradual",
    ),
    "GDS02": (
        "Datasets.Synthetic2.Gradual.GDS02",
        "get_GDS02",
        "gradual",
    ),
    "GDS03": (
        "Datasets.Synthetic2.Gradual.GDS03",
        "get_GDS03",
        "gradual",
    ),
    "GDS04": (
        "Datasets.Synthetic2.Gradual.GDS04",
        "get_GDS04",
        "gradual",
    ),
    "GDS05": (
        "Datasets.Synthetic2.Gradual.GDS05",
        "get_GDS05",
        "gradual",
    ),
    "GDS06": (
        "Datasets.Synthetic2.Gradual.GDS06",
        "get_GDS06",
        "gradual",
    ),
}


MODEL_LABELS = {
    "OLR-WA": "OLR-WA-SCCM",
    "PA": "PA-SCCM",
    "RLS": "RLS-SCCM",
    "WidrowHoff": "WidrowHoff-SCCM",
}


def safe_float(value):
    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except Exception:
        return None


def is_truthy(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return value != 0

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def write_csv(rows, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        file_path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({
        key
        for row in rows
        for key in row.keys()
    })

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(file_path):
    file_path = Path(file_path)

    if (
        not file_path.exists()
        or file_path.stat().st_size == 0
    ):
        return []

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


class SCCMConsoleParser:
    MINI_BATCH_RE = re.compile(
        r"mini-batch-\s*(\d+)"
    )

    ADDITIONAL_RE = re.compile(
        r"additional mini-batch request #\s*(\d+)"
    )

    DRIFT_MAG_RE = re.compile(
        r"drift_magnitude\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    def __init__(
        self,
        model_label,
        dataset,
        drift_type,
        index_mode,
        increment_size=1,
        base_offset=0,
    ):
        self.model_label = model_label
        self.dataset = dataset
        self.drift_type = drift_type
        self.index_mode = index_mode
        self.increment_size = int(increment_size)
        self.base_offset = int(base_offset)

        self.current_iteration = None
        self.pending_drift_magnitude = None
        self.last_event = None
        self.events = []

    def alarm_index(self, iteration):
        iteration = int(iteration)

        if (
            self.index_mode
            == "base_plus_iteration_times_increment"
        ):
            return (
                self.base_offset
                + iteration * self.increment_size
            )

        if (
            self.index_mode
            == "zero_based_iteration_plus_one"
        ):
            return (
                iteration + 1
            ) * self.increment_size

        return iteration

    def parse_line(self, line):
        line = line.strip()

        if not line:
            return

        mini_match = self.MINI_BATCH_RE.search(line)

        if mini_match:
            self.current_iteration = int(
                mini_match.group(1)
            )

        if "drift_magnitude" in line:
            matches = self.DRIFT_MAG_RE.findall(line)

            if matches:
                self.pending_drift_magnitude = safe_float(
                    matches[-1]
                )

        if (
            "SHORT TERM DRIFT DETECTED" in line
            and "True" in line
        ):
            if self.current_iteration is None:
                return

            alarm_index = int(
                self.alarm_index(
                    self.current_iteration
                )
            )

            event = {
                "model": self.model_label,
                "dataset": self.dataset,
                "drift_type": self.drift_type,
                "iteration": int(
                    self.current_iteration
                ),
                "alarm_index": alarm_index,
                "short_term_drift_detected": True,
                "long_term_drift_detected": False,
                "recalibration_triggered": False,
                "recalibration_batches": 0,
                "adaptation_triggered": True,
                "adaptation_applied": True,
                "drift_magnitude":
                    self.pending_drift_magnitude,
                "tuned_hyperparameter": "",
                "tuned_hyperparameter_value": None,
            }

            self.events.append(event)
            self.last_event = event
            return

        if (
            self.last_event is not None
            and line.startswith("tuned_w_inc")
        ):
            self.last_event[
                "tuned_hyperparameter"
            ] = "w_inc"

            self.last_event[
                "tuned_hyperparameter_value"
            ] = safe_float(line.split()[-1])

            return

        if (
            self.last_event is not None
            and line.startswith("tuned_C")
        ):
            self.last_event[
                "tuned_hyperparameter"
            ] = "C"

            self.last_event[
                "tuned_hyperparameter_value"
            ] = safe_float(line.split()[-1])

            return

        if (
            self.last_event is not None
            and line.startswith("tuned_lambda")
        ):
            self.last_event[
                "tuned_hyperparameter"
            ] = "lambda"

            self.last_event[
                "tuned_hyperparameter_value"
            ] = safe_float(line.split()[-1])

            return

        if (
            self.last_event is not None
            and line.startswith(
                "tuned_learning_rate"
            )
        ):
            self.last_event[
                "tuned_hyperparameter"
            ] = "learning_rate"

            self.last_event[
                "tuned_hyperparameter_value"
            ] = safe_float(line.split()[-1])

            return

        if (
            self.last_event is not None
            and "Long Term Drift Detected" in line
        ):
            is_long_term = "True" in line

            self.last_event[
                "long_term_drift_detected"
            ] = is_long_term

            if is_long_term:
                self.last_event[
                    "recalibration_triggered"
                ] = True

            return

        if self.last_event is not None:
            additional_match = (
                self.ADDITIONAL_RE.search(line)
            )

            if additional_match:
                self.last_event[
                    "recalibration_triggered"
                ] = True

                self.last_event[
                    "recalibration_batches"
                ] = max(
                    int(
                        self.last_event.get(
                            "recalibration_batches",
                            0,
                        )
                        or 0
                    ),
                    int(additional_match.group(1)),
                )


class ParsingTee(io.TextIOBase):

    def __init__(
        self,
        console_stream,
        log_stream,
        parser,
    ):
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
            line, self.buffer = self.buffer.split(
                "\n",
                1,
            )
            self.parser.parse_line(line)

        return len(data)

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()

        if self.buffer:
            self.parser.parse_line(self.buffer)
            self.buffer = ""


def load_dataset(dataset_name, seed):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    module_name, getter_name, drift_type = (
        DATASET_REGISTRY[dataset_name]
    )

    module = importlib.import_module(module_name)
    getter = getattr(module, getter_name)

    output = getter(
        seed=seed,
        return_meta=True,
    )

    X = output[0]
    y = output[1]
    meta = output[-1]

    return X, y, meta, drift_type


def get_true_drift_points(meta, drift_type):
    if drift_type == "abrupt":
        return [
            int(meta.get("drift_point"))
        ]

    if drift_type == "incremental":
        step = int(
            meta.get("samples_per_step")
        )

        n_steps = int(
            meta.get("n_steps")
        )

        return [
            step * index
            for index in range(1, n_steps)
        ]

    if drift_type == "gradual":
        segment_lengths = [
            int(value)
            for value in meta.get(
                "segment_lengths",
                [],
            )
        ]

        points = []
        running = 0

        for length in segment_lengths[:-1]:
            running += length
            points.append(running)

        return points

    return []


def select_candidate_events(event_log, source):
    if source == "short_term":
        return [
            event
            for event in event_log
            if is_truthy(
                event.get(
                    "short_term_drift_detected"
                )
            )
        ]

    if source == "long_term":
        return [
            event
            for event in event_log
            if is_truthy(
                event.get(
                    "long_term_drift_detected"
                )
            )
        ]

    if source == "recalibration":
        return [
            event
            for event in event_log
            if is_truthy(
                event.get(
                    "recalibration_triggered"
                )
            )
        ]

    raise ValueError(
        f"Unknown candidate_source: {source}"
    )


def consolidate_alarm_episodes(
    candidate_events,
    cooldown,
):
    sorted_events = sorted(
        candidate_events,
        key=lambda event: int(
            event["alarm_index"]
        ),
    )

    episodes = []
    current = None
    current_end = -1

    for event in sorted_events:
        alarm_index = int(
            event["alarm_index"]
        )

        if (
            current is None
            or alarm_index > current_end
        ):
            current = {
                "episode_alarm_index":
                    alarm_index,
                "episode_start_index":
                    alarm_index,
                "episode_end_index":
                    alarm_index,
                "episode_size": 1,
                "episode_alarm_indices": [
                    alarm_index
                ],
            }

            episodes.append(current)

            current_end = (
                alarm_index
                + int(cooldown)
            )

        else:
            current[
                "episode_end_index"
            ] = alarm_index

            current["episode_size"] += 1

            current[
                "episode_alarm_indices"
            ].append(alarm_index)

    return episodes


def filter_episodes_by_size(
    episodes,
    min_episode_size,
):
    min_episode_size = int(
        min_episode_size
    )

    if min_episode_size <= 1:
        return episodes

    return [
        episode
        for episode in episodes
        if int(
            episode.get(
                "episode_size",
                1,
            )
        ) >= min_episode_size
    ]


def match_episodes_to_true_drifts(
    episodes,
    true_drift_points,
    tolerance,
    model="",
    dataset="",
    seed="",
    drift_type="",
):
    """Match retained alarm episodes to true drifts chronologically.

    An episode is eligible only when its first alarm event falls inside the
    post-drift window [d, d + tolerance]. Later events cannot convert an episode
    that began before the drift into a true positive. Each drift and episode is
    matched at most once, chronologically.
    """
    normalized_episodes = []
    for episode in episodes:
        alarm_indices = sorted(
            int(value)
            for value in (
                episode.get("episode_alarm_indices")
                or [episode["episode_alarm_index"]]
            )
        )
        if not alarm_indices:
            continue

        item = dict(episode)
        item["episode_alarm_indices"] = alarm_indices
        item["episode_alarm_index"] = alarm_indices[0]
        item["episode_start_index"] = int(
            episode.get("episode_start_index", alarm_indices[0])
        )
        item["episode_end_index"] = int(
            episode.get("episode_end_index", alarm_indices[-1])
        )
        item["episode_size"] = int(
            episode.get("episode_size", len(alarm_indices))
        )
        normalized_episodes.append(item)

    normalized_episodes.sort(
        key=lambda item: (
            int(item["episode_start_index"]),
            int(item["episode_end_index"]),
        )
    )

    used_episode_ids = set()
    delays = []
    detail_rows = []
    matched_drift_points = []
    matched_alarm_indices = []
    matched_episode_starts = []
    matched_episode_alarm_lists = []

    tp = 0
    fn = 0

    for drift_point in sorted(int(value) for value in true_drift_points):
        tolerance_start = drift_point
        tolerance_end = drift_point + int(tolerance)
        nearby_episode_ids = []
        nearby_episode_starts = []
        nearby_episode_alarm_lists = []
        matched_episode_id = None

        for episode_id, episode in enumerate(normalized_episodes):
            episode_alarm_index = int(episode["episode_alarm_index"])
            if not (tolerance_start <= episode_alarm_index <= tolerance_end):
                continue

            nearby_episode_ids.append(episode_id)
            nearby_episode_starts.append(int(episode["episode_start_index"]))
            nearby_episode_alarm_lists.append(
                ";".join(map(str, episode["episode_alarm_indices"]))
            )

            if matched_episode_id is None and episode_id not in used_episode_ids:
                matched_episode_id = episode_id

        common = {
            "evaluation_level": "episode_first_trigger_primary",
            "matching_rule": "episode_first_trigger_in_post_drift_window",
            "row_type": "true_drift",
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "drift_type": drift_type,
            "true_drift_point": drift_point,
            "tolerance_start": tolerance_start,
            "tolerance_end": tolerance_end,
            "nearby_alarm_episode_count": len(nearby_episode_ids),
            "nearby_episode_starts": ";".join(map(str, nearby_episode_starts)),
            "nearby_episode_alarm_lists": " | ".join(nearby_episode_alarm_lists),
        }

        if matched_episode_id is None:
            fn += 1
            detail_rows.append({
                **common,
                "episode_start": "",
                "episode_end": "",
                "episode_size": "",
                "episode_alarm_indices": "",
                "matched_alarm_index": "",
                "delay_samples": "",
                "status": "FN",
            })
            continue

        tp += 1
        used_episode_ids.add(matched_episode_id)
        episode = normalized_episodes[matched_episode_id]
        matched_alarm_index = int(episode["episode_alarm_index"])
        episode_start = int(episode["episode_start_index"])
        delay = matched_alarm_index - drift_point

        delays.append(delay)
        matched_drift_points.append(drift_point)
        matched_alarm_indices.append(matched_alarm_index)
        matched_episode_starts.append(episode_start)
        matched_episode_alarm_lists.append(
            ";".join(map(str, episode["episode_alarm_indices"]))
        )
        detail_rows.append({
            **common,
            "episode_start": episode_start,
            "episode_end": int(episode["episode_end_index"]),
            "episode_size": int(episode.get("episode_size", 1)),
            "episode_alarm_indices": ";".join(
                map(str, episode["episode_alarm_indices"])
            ),
            "matched_alarm_index": matched_alarm_index,
            "delay_samples": delay,
            "status": "TP",
        })

    for episode_id, episode in enumerate(normalized_episodes):
        if episode_id in used_episode_ids:
            continue
        detail_rows.append({
            "evaluation_level": "episode_first_trigger_primary",
            "matching_rule": "episode_first_trigger_in_post_drift_window",
            "row_type": "false_positive_episode",
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "drift_type": drift_type,
            "true_drift_point": "",
            "tolerance_start": "",
            "tolerance_end": "",
            "nearby_alarm_episode_count": "",
            "nearby_episode_starts": "",
            "nearby_episode_alarm_lists": "",
            "episode_start": int(episode["episode_start_index"]),
            "episode_end": int(episode["episode_end_index"]),
            "episode_size": int(episode.get("episode_size", 1)),
            "episode_alarm_indices": ";".join(
                map(str, episode["episode_alarm_indices"])
            ),
            "matched_alarm_index": "",
            "delay_samples": "",
            "status": "FP",
        })

    fp = len(normalized_episodes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_delay = sum(delays) / len(delays) if delays else None
    median_delay = None
    if delays:
        ordered = sorted(delays)
        middle = len(ordered) // 2
        median_delay = (
            float(ordered[middle])
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay": mean_delay,
        "median_delay": median_delay,
        "delay_sum": sum(delays),
        "delay_count": len(delays),
        "matched_drift_points": ";".join(map(str, matched_drift_points)),
        "matched_alarm_indices": ";".join(map(str, matched_alarm_indices)),
        "matched_episode_starts": ";".join(map(str, matched_episode_starts)),
        "matched_episode_alarm_lists": " | ".join(matched_episode_alarm_lists),
        "detail_rows": detail_rows,
    }

def match_alarm_indices_to_true_drifts(
    alarm_indices,
    true_drift_points,
    tolerance,
    model="",
    dataset="",
    seed="",
    drift_type="",
    evaluation_level="raw_trigger_supplemental",
):
    """Chronologically match individual alarms to true drifts one-to-one."""
    alarms = sorted(int(value) for value in alarm_indices)
    used_alarm_ids = set()
    delays = []
    detail_rows = []
    matched_drift_points = []
    matched_alarm_indices = []
    tp = 0
    fn = 0

    for drift_point in sorted(int(value) for value in true_drift_points):
        tolerance_start = drift_point
        tolerance_end = drift_point + int(tolerance)
        nearby_ids = [
            alarm_id
            for alarm_id, alarm_index in enumerate(alarms)
            if tolerance_start <= alarm_index <= tolerance_end
        ]
        matched_alarm_id = next(
            (alarm_id for alarm_id in nearby_ids if alarm_id not in used_alarm_ids),
            None,
        )
        common = {
            "evaluation_level": evaluation_level,
            "matching_rule": "individual_alarm",
            "row_type": "true_drift",
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "drift_type": drift_type,
            "true_drift_point": drift_point,
            "tolerance_start": tolerance_start,
            "tolerance_end": tolerance_end,
            "nearby_raw_alarm_count": len(nearby_ids),
            "nearby_raw_alarm_indices": ";".join(str(alarms[i]) for i in nearby_ids),
        }
        if matched_alarm_id is None:
            fn += 1
            detail_rows.append({
                **common,
                "matched_alarm_index": "",
                "delay_samples": "",
                "status": "FN",
            })
            continue

        tp += 1
        used_alarm_ids.add(matched_alarm_id)
        matched_alarm = alarms[matched_alarm_id]
        delay = matched_alarm - drift_point
        delays.append(delay)
        matched_drift_points.append(drift_point)
        matched_alarm_indices.append(matched_alarm)
        detail_rows.append({
            **common,
            "matched_alarm_index": matched_alarm,
            "delay_samples": delay,
            "status": "TP",
        })

    for alarm_id, alarm_index in enumerate(alarms):
        if alarm_id in used_alarm_ids:
            continue
        detail_rows.append({
            "evaluation_level": evaluation_level,
            "matching_rule": "individual_alarm",
            "row_type": "false_positive_alarm",
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "drift_type": drift_type,
            "true_drift_point": "",
            "tolerance_start": "",
            "tolerance_end": "",
            "nearby_raw_alarm_count": "",
            "nearby_raw_alarm_indices": "",
            "matched_alarm_index": alarm_index,
            "delay_samples": "",
            "status": "FP",
        })

    fp = len(alarms) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_delay = sum(delays) / len(delays) if delays else None
    median_delay = None
    if delays:
        ordered = sorted(delays)
        middle = len(ordered) // 2
        median_delay = (
            float(ordered[middle])
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay": mean_delay,
        "median_delay": median_delay,
        "delay_sum": sum(delays),
        "delay_count": len(delays),
        "matched_drift_points": ";".join(map(str, matched_drift_points)),
        "matched_alarm_indices": ";".join(map(str, matched_alarm_indices)),
        "detail_rows": detail_rows,
    }


def summarize_seed(
    event_log,
    config,
    meta,
    drift_type,
    n_samples,
    processing_increment=1,
    processed_samples=None,
    resource_measurement=None,
):
    true_drift_points = get_true_drift_points(meta, drift_type)
    tolerance = int(round(float(config.get("tolerance_ratio", 0.05)) * n_samples))
    cooldown = int(round(float(config.get("cooldown_factor", 2.0)) * tolerance))
    min_episode_size = int(config.get("min_episode_size", 2))
    candidate_source = config.get("candidate_source", "long_term")
    candidates = select_candidate_events(event_log, candidate_source)
    raw_alarm_indices = [int(event["alarm_index"]) for event in candidates]

    raw_episodes = consolidate_alarm_episodes(candidates, cooldown)
    episodes = filter_episodes_by_size(raw_episodes, min_episode_size)
    model_label = MODEL_LABELS.get(config["model"], config["model"])
    seed = config.get("current_seed", "")

    episode_match = match_episodes_to_true_drifts(
        episodes,
        true_drift_points,
        tolerance,
        model=model_label,
        dataset=config["dataset"],
        seed=seed,
        drift_type=drift_type,
    )
    raw_match = match_alarm_indices_to_true_drifts(
        raw_alarm_indices,
        true_drift_points,
        tolerance,
        model=model_label,
        dataset=config["dataset"],
        seed=seed,
        drift_type=drift_type,
        evaluation_level="raw_trigger_supplemental",
    )

    processing_increment = max(1, int(processing_increment or 1))
    processed_samples = int(processed_samples if processed_samples is not None else n_samples)

    def delay_metrics(match):
        mean_samples = match["mean_delay"]
        median_samples = match["median_delay"]
        mean_increments = (
            mean_samples / processing_increment if mean_samples is not None else None
        )
        median_increments = (
            median_samples / processing_increment if median_samples is not None else None
        )
        practical = (
            max(0.0, mean_increments - 1.0) if mean_increments is not None else None
        )
        return mean_samples, median_samples, mean_increments, median_increments, practical

    mean_samples, median_samples, mean_increments, median_increments, practical = delay_metrics(episode_match)
    raw_mean_samples, raw_median_samples, raw_mean_increments, raw_median_increments, _ = delay_metrics(raw_match)

    adaptations = sum(
        1 for event in event_log if is_truthy(event.get("adaptation_triggered"))
    )
    recalibrations = sum(
        1 for event in event_log if is_truthy(event.get("recalibration_triggered"))
    )
    recalibration_batches = sum(
        int(event.get("recalibration_batches", 0) or 0) for event in event_log
    )
    interventions = adaptations + recalibrations

    episode_lists = [
        ";".join(map(str, episode.get("episode_alarm_indices", [])))
        for episode in episodes
    ]
    raw_episode_lists = [
        ";".join(map(str, episode.get("episode_alarm_indices", [])))
        for episode in raw_episodes
    ]

    measurement = resource_measurement.as_dict() if resource_measurement is not None else {}
    runtime_seconds = float(measurement.get("runtime_seconds", 0.0) or 0.0)

    summary = {
        "row_type": "seed_result",
        "method_family": "SCCM",
        "primary_evaluation": "episode_first_trigger_in_window",
        "supplemental_evaluation": "raw_trigger",
        "model": model_label,
        "base_model": config["model"],
        "method": model_label,
        "dataset": config["dataset"],
        "drift_type": drift_type,
        "seed": seed,
        "seed_count": 1,
        "true_drifts": len(true_drift_points),
        "true_drift_points": ";".join(map(str, true_drift_points)),
        "candidate_source": candidate_source,
        "candidate_triggers": len(candidates),
        "raw_candidate_confirmations": len(candidates),
        "raw_alarm_episodes": len(raw_episodes),
        "alarm_episodes": len(episodes),
        "removed_small_episodes": max(0, len(raw_episodes) - len(episodes)),
        "duplicate_candidate_triggers": max(0, len(candidates) - len(raw_episodes)),
        "episode_alarm_lists": " | ".join(episode_lists),
        "raw_episode_alarm_lists": " | ".join(raw_episode_lists),
        "episode_sizes": ";".join(str(episode.get("episode_size", 1)) for episode in episodes),
        "raw_episode_sizes": ";".join(str(episode.get("episode_size", 1)) for episode in raw_episodes),
        "tolerance": tolerance,
        "tolerance_ratio": config.get("tolerance_ratio", 0.05),
        "cooldown": cooldown,
        "cooldown_factor": config.get("cooldown_factor", 2.0),
        "min_episode_size": min_episode_size,
        "multiplier": config.get("multiplier", ""),
        "kpi": config.get("kpi", ""),
        "sccm_window_size": config.get("sccm_window_size", DEFAULT_SCCM_WINDOW_SIZE),
        "used_kpi_window_size": config.get("used_kpi_window_size", DEFAULT_USED_KPI_WINDOW_SIZE),
        "increment_user_value": config.get("increment_user_value", ""),
        "processing_increment": processing_increment,
        "processed_samples": processed_samples,
        "adaptations": adaptations,
        "recalibrations": recalibrations,
        "recalibration_batches": recalibration_batches,
        "interventions": interventions,
        "adaptations_per_1000": per_1000(adaptations, processed_samples),
        "recalibrations_per_1000": per_1000(recalibrations, processed_samples),
        "interventions_per_1000": per_1000(interventions, processed_samples),
        "runtime_seconds": runtime_seconds,
        "runtime_per_1000_samples": per_1000(runtime_seconds, processed_samples),
        "rss_before_mb": measurement.get("rss_before_mb", ""),
        "peak_rss_mb": measurement.get("peak_rss_mb", ""),
        "peak_rss_delta_mb": measurement.get("peak_rss_delta_mb", ""),
        "memory_measurement_method": measurement.get("memory_measurement_method", ""),
    }

    summary.update({
        "tp": episode_match["tp"],
        "fp": episode_match["fp"],
        "fn": episode_match["fn"],
        "precision": round(episode_match["precision"], 8),
        "recall": round(episode_match["recall"], 8),
        "f1": round(episode_match["f1"], 8),
        "mean_delay_samples": round(mean_samples, 6) if mean_samples is not None else "",
        "median_delay_samples": round(median_samples, 6) if median_samples is not None else "",
        "mean_delay_increments": round(mean_increments, 6) if mean_increments is not None else "",
        "median_delay_increments": round(median_increments, 6) if median_increments is not None else "",
        "mean_delay": round(mean_samples, 6) if mean_samples is not None else "",
        "median_delay": round(median_samples, 6) if median_samples is not None else "",
        "mean_delay_batches": round(mean_increments, 6) if mean_increments is not None else "",
        "median_delay_batches": round(median_increments, 6) if median_increments is not None else "",
        "practical_delay_batches": round(practical, 6) if practical is not None else "",
        "delay_sum": episode_match["delay_sum"],
        "delay_count": episode_match["delay_count"],
        "matched_drift_points": episode_match["matched_drift_points"],
        "matched_alarm_indices": episode_match["matched_alarm_indices"],
        "matched_episode_starts": episode_match["matched_episode_starts"],
        "matched_episode_alarm_lists": episode_match["matched_episode_alarm_lists"],
        "raw_tp": raw_match["tp"],
        "raw_fp": raw_match["fp"],
        "raw_fn": raw_match["fn"],
        "raw_precision": round(raw_match["precision"], 8),
        "raw_recall": round(raw_match["recall"], 8),
        "raw_f1": round(raw_match["f1"], 8),
        "raw_mean_delay_samples": round(raw_mean_samples, 6) if raw_mean_samples is not None else "",
        "raw_median_delay_samples": round(raw_median_samples, 6) if raw_median_samples is not None else "",
        "raw_mean_delay_increments": round(raw_mean_increments, 6) if raw_mean_increments is not None else "",
        "raw_median_delay_increments": round(raw_median_increments, 6) if raw_median_increments is not None else "",
        "raw_delay_sum": raw_match["delay_sum"],
        "raw_delay_count": raw_match["delay_count"],
    })

    detail_rows = episode_match["detail_rows"] + raw_match["detail_rows"]
    return summary, detail_rows

def aggregate_seed_summaries(
    seed_summaries,
    config,
    drift_type,
):
    """Create a convenience pooled row while preserving seed rows for inference."""
    count_fields = [
        "true_drifts", "candidate_triggers", "raw_candidate_confirmations",
        "raw_alarm_episodes", "alarm_episodes", "removed_small_episodes",
        "duplicate_candidate_triggers", "tp", "fp", "fn", "raw_tp",
        "raw_fp", "raw_fn", "adaptations", "recalibrations",
        "recalibration_batches", "interventions", "processed_samples",
    ]
    totals = {
        field: sum(int(float(row.get(field, 0) or 0)) for row in seed_summaries)
        for field in count_fields
    }

    def micro(prefix=""):
        tp = totals[f"{prefix}tp"]
        fp = totals[f"{prefix}fp"]
        fn = totals[f"{prefix}fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    precision, recall, f1 = micro("")
    raw_precision, raw_recall, raw_f1 = micro("raw_")

    def weighted_delay(sum_field, count_field):
        total_sum = sum(float(row.get(sum_field, 0) or 0) for row in seed_summaries)
        total_count = sum(int(float(row.get(count_field, 0) or 0)) for row in seed_summaries)
        return total_sum, total_count, (total_sum / total_count if total_count else "")

    delay_sum, delay_count, mean_delay_samples = weighted_delay("delay_sum", "delay_count")
    raw_delay_sum, raw_delay_count, raw_mean_delay_samples = weighted_delay("raw_delay_sum", "raw_delay_count")

    total_runtime = sum(float(row.get("runtime_seconds", 0) or 0) for row in seed_summaries)
    processed_samples = totals["processed_samples"]
    processing_increment_values = {
        int(float(row.get("processing_increment", 1) or 1)) for row in seed_summaries
    }
    processing_increment = next(iter(processing_increment_values)) if len(processing_increment_values) == 1 else ""
    mean_delay_increments = (
        mean_delay_samples / processing_increment
        if mean_delay_samples != "" and processing_increment not in ("", 0)
        else ""
    )
    raw_mean_delay_increments = (
        raw_mean_delay_samples / processing_increment
        if raw_mean_delay_samples != "" and processing_increment not in ("", 0)
        else ""
    )

    output = {
        "row_type": "aggregate_for_dataset",
        "method_family": "SCCM",
        "primary_evaluation": "episode_first_trigger_in_window",
        "supplemental_evaluation": "raw_trigger",
        "model": MODEL_LABELS.get(config["model"], config["model"]),
        "base_model": config["model"],
        "method": MODEL_LABELS.get(config["model"], config["model"]),
        "dataset": config["dataset"],
        "drift_type": drift_type,
        "seed_count": len(seed_summaries),
        "seeds": ";".join(str(seed) for seed in config.get("seeds", DEFAULT_SEEDS)),
        "candidate_source": config.get("candidate_source", "long_term"),
        "tolerance_ratio": config.get("tolerance_ratio", 0.05),
        "cooldown_factor": config.get("cooldown_factor", 2.0),
        "min_episode_size": config.get("min_episode_size", 2),
        "multiplier": config.get("multiplier", ""),
        "kpi": config.get("kpi", ""),
        "sccm_window_size": config.get("sccm_window_size", DEFAULT_SCCM_WINDOW_SIZE),
        "used_kpi_window_size": config.get("used_kpi_window_size", DEFAULT_USED_KPI_WINDOW_SIZE),
        "increment_user_value": config.get("increment_user_value", ""),
        "processing_increment": processing_increment,
        **totals,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "raw_precision": round(raw_precision, 6),
        "raw_recall": round(raw_recall, 6),
        "raw_f1": round(raw_f1, 6),
        "delay_sum": delay_sum,
        "delay_count": delay_count,
        "mean_delay_samples": round(mean_delay_samples, 6) if mean_delay_samples != "" else "",
        "mean_delay_increments": round(mean_delay_increments, 6) if mean_delay_increments != "" else "",
        "mean_delay": round(mean_delay_samples, 6) if mean_delay_samples != "" else "",
        "mean_delay_batches": round(mean_delay_increments, 6) if mean_delay_increments != "" else "",
        "practical_delay_batches": (
            round(max(0.0, mean_delay_increments - 1.0), 6)
            if mean_delay_increments != "" else ""
        ),
        "raw_delay_sum": raw_delay_sum,
        "raw_delay_count": raw_delay_count,
        "raw_mean_delay_samples": round(raw_mean_delay_samples, 6) if raw_mean_delay_samples != "" else "",
        "raw_mean_delay_increments": round(raw_mean_delay_increments, 6) if raw_mean_delay_increments != "" else "",
        "runtime_seconds": total_runtime,
        "runtime_per_1000_samples": per_1000(total_runtime, processed_samples),
        "adaptations_per_1000": per_1000(totals["adaptations"], processed_samples),
        "recalibrations_per_1000": per_1000(totals["recalibrations"], processed_samples),
        "interventions_per_1000": per_1000(totals["interventions"], processed_samples),
        "peak_rss_mb": max(float(row.get("peak_rss_mb", 0) or 0) for row in seed_summaries),
        "peak_rss_delta_mb": max(float(row.get("peak_rss_delta_mb", 0) or 0) for row in seed_summaries),
        "memory_measurement_method": next((row.get("memory_measurement_method", "") for row in seed_summaries if row.get("memory_measurement_method")), ""),
    }
    return output

def build_parser(
    config,
    n_features,
    n_samples,
):
    model = config["model"]

    model_label = MODEL_LABELS.get(
        model,
        model,
    )

    dataset = config["dataset"]

    drift_type = DATASET_REGISTRY[
        dataset
    ][2]

    if model == "OLR-WA":
        increment_size = (
            Hyperparameter
            .olr_wa_increment_size(
                n_features,
                user_defined_val=int(
                    config.get(
                        "increment_user_value",
                        10,
                    )
                ),
            )
        )

        base_offset = (
            Util
            .calculate_no_of_base_model_points(
                n_samples,
                Hyperparameter
                .olr_wa_base_model_size0,
            )
        )

        parser = SCCMConsoleParser(
            model_label,
            dataset,
            drift_type,
            "base_plus_iteration_times_increment",
            increment_size,
            base_offset,
        )

        return (
            parser,
            increment_size,
            base_offset,
        )

    if model == "WidrowHoff":
        increment_size = int(
            config.get(
                "report_interval",
                1,
            )
        )

        base_model_size = int(
            config.get(
                "report_interval",
                1,
            )
        )

        base_offset = (
            Util
            .calculate_no_of_base_model_points(
                n_samples,
                base_model_size,
            )
        )

        parser = SCCMConsoleParser(
            model_label,
            dataset,
            drift_type,
            "base_plus_iteration_times_increment",
            increment_size,
            base_offset,
        )

        return (
            parser,
            increment_size,
            base_offset,
        )

    if model in {"PA", "RLS"}:
        increment_size = 1
        base_offset = 0

        parser = SCCMConsoleParser(
            model_label,
            dataset,
            drift_type,
            "zero_based_iteration_plus_one",
            increment_size,
            base_offset,
        )

        return (
            parser,
            increment_size,
            base_offset,
        )

    raise ValueError(
        f"Unknown model: {model}"
    )


def call_model(
    config,
    X_train,
    y_train,
    X_test,
    y_test,
):
    model = config["model"]

    multiplier = float(
        config.get(
            "multiplier",
            1.5,
        )
    )

    kpi = config.get(
        "kpi",
        "MSE",
    )

    dataset = config.get(
        "scale_ds",
        config["dataset"],
    )

    sccm_window_size = int(
        config.get(
            "sccm_window_size",
            DEFAULT_SCCM_WINDOW_SIZE,
        )
    )

    used_kpi_window_size = int(
        config.get(
            "used_kpi_window_size",
            DEFAULT_USED_KPI_WINDOW_SIZE,
        )
    )

    if model == "OLR-WA":
        number_of_features = (
            X_train.shape[1]
        )

        increment_default = (
            Hyperparameter
            .olr_wa_increment_size(
                number_of_features,
                user_defined_val=int(
                    config.get(
                        "increment_user_value",
                        10,
                    )
                ),
            )
        )

        increment_size = config.get(
            "increment_user_value",
            increment_default,
        )

        return OLR_WA_SCCM.olr_wa_sccm(
            X_train,
            y_train,
            Hyperparameter.olr_wa_w_base,
            Hyperparameter.olr_wa_w_inc,
            Hyperparameter
            .olr_wa_base_model_size0,
            increment_size,
            X_test,
            y_test,
            kpi=kpi,
            multiplier=multiplier,
            sccm_window_size=
                sccm_window_size,
            used_kpi_window_size=
                used_kpi_window_size,
        )

    if model == "PA":
        return PA_SCCM.ad_pa_generic(
            X_train,
            y_train,
            float(
                config.get(
                    "pa_c",
                    Hyperparameter.pa_C,
                )
            ),
            float(
                config.get(
                    "pa_epsilon",
                    Hyperparameter
                    .pa_epsilon,
                )
            ),
            X_test,
            y_test,
            kpi=kpi,
            multiplier=multiplier,
            report_interval=int(
                config.get(
                    "report_interval",
                    10,
                )
            ),
            ds=dataset,
            c_bounds=tuple(
                config.get(
                    "pa_c_bounds",
                    (0.1, 10.0),
                )
            ),
            sccm_window_size=
                sccm_window_size,
            used_kpi_window_size=
                used_kpi_window_size,
        )

    if model == "RLS":
        return RLS_SCCM.ad_rls_generic(
            X_train,
            y_train,
            float(
                config.get(
                    "rls_lambda",
                    Hyperparameter
                    .rls_lambda_,
                )
            ),
            float(
                config.get(
                    "rls_delta",
                    Hyperparameter
                    .rls_delta,
                )
            ),
            X_test,
            y_test,
            kpi=kpi,
            multiplier=multiplier,
            DS=dataset,
            report_interval=int(
                config.get(
                    "report_interval",
                    10,
                )
            ),
            lambda_bounds=tuple(
                config.get(
                    "rls_lambda_bounds",
                    (0.85, 0.999),
                )
            ),
            sccm_window_size=
                sccm_window_size,
            used_kpi_window_size=
                used_kpi_window_size,
        )

    if model == "WidrowHoff":
        return (
            WidrowHoff_SCCM
            .ad_widrow_hoff_generic(
                X_train,
                y_train,
                float(
                    config.get(
                        "wh_learning_rate",
                        Hyperparameter
                        .wf_learning_rate,
                    )
                ),
                X_test,
                y_test,
                kpi=kpi,
                multiplier=multiplier,
                DS=dataset,
                report_interval=int(
                    config.get(
                        "report_interval",
                        1,
                    )
                ),
                sccm_window_size=
                    sccm_window_size,
                used_kpi_window_size=
                    used_kpi_window_size,
            )
        )

    raise ValueError(
        f"Unknown model: {model}"
    )


def run_one_seed(
    config,
    seed,
    log_file,
):
    X, y, meta, drift_type = load_dataset(config["dataset"], seed)
    number_of_samples = int(X.shape[0])
    train_size = int(float(config.get("train_percent", 90)) * number_of_samples / 100)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    parser, processing_increment, base_offset = build_parser(
        config, X_train.shape[1], number_of_samples
    )
    print(f"\nRunning {config['model']} on {config['dataset']} seed={seed}")
    print("drift_type:", drift_type)
    true_drift_points = get_true_drift_points(meta, drift_type)
    print("true_drift_points:", true_drift_points)
    print("Total Original Drifts in Dataset:", len(true_drift_points))
    print("n_samples:", number_of_samples)
    print("processed_stream_samples:", train_size)
    print("processing_increment:", processing_increment)
    print("base_offset:", base_offset)
    print("config:", {key: value for key, value in config.items() if key != "current_seed"})

    tee = ParsingTee(sys.stdout, log_file, parser)
    with redirect_stdout(tee), redirect_stderr(tee):
        _, resource_measurement = measure_call(
            call_model,
            config,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    tee.flush()

    for event in parser.events:
        event["seed"] = seed
        event["true_drift_points"] = ";".join(map(str, true_drift_points))
        event["train_percent"] = config.get("train_percent", 90)
        event["multiplier"] = config.get("multiplier", "")
        event["kpi"] = config.get("kpi", "")
        event["sccm_window_size"] = config.get("sccm_window_size", DEFAULT_SCCM_WINDOW_SIZE)
        event["used_kpi_window_size"] = config.get("used_kpi_window_size", DEFAULT_USED_KPI_WINDOW_SIZE)
        event["processing_increment"] = processing_increment
        event["processed_samples"] = train_size

    return (
        parser.events,
        meta,
        drift_type,
        number_of_samples,
        processing_increment,
        train_size,
        resource_measurement,
    )

def run_quality_experiment(
    config,
    script_file=None,
):
    config = dict(config)

    config.setdefault(
        "seeds",
        DEFAULT_SEEDS,
    )

    config.setdefault(
        "train_percent",
        90,
    )

    config.setdefault(
        "candidate_source",
        "long_term",
    )

    config.setdefault(
        "cooldown_factor",
        1.0,
    )

    config.setdefault(
        "tolerance_ratio",
        0.05,
    )

    config.setdefault(
        "min_episode_size",
        1,
    )

    config.setdefault(
        "multiplier",
        1.5,
    )

    config.setdefault(
        "kpi",
        "MSE",
    )

    config.setdefault(
        "sccm_window_size",
        DEFAULT_SCCM_WINDOW_SIZE,
    )

    config.setdefault(
        "used_kpi_window_size",
        DEFAULT_USED_KPI_WINDOW_SIZE,
    )

    config.setdefault(
        "increment_user_value",
        10,
    )

    if script_file is None:
        script_file = __file__

    output_directory = (
        Path(script_file)
        .resolve()
        .parent
        / "quality_outputs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        config["model"]
        .replace("-", "")
        .replace(" ", "")
        + "_"
        + config["dataset"]
    )

    events_csv = (
        output_directory
        / f"{stem}_events.csv"
    )

    summary_csv = (
        output_directory
        / f"{stem}_summary.csv"
    )

    drift_points_csv = (
        output_directory
        / f"{stem}_drift_point_details.csv"
    )

    errors_csv = (
        output_directory
        / f"{stem}_errors.csv"
    )

    console_log = (
        output_directory
        / f"{stem}_console.log"
    )

    all_events = []
    all_drift_point_rows = []
    seed_summaries = []
    error_rows = []
    drift_type = None

    with console_log.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        def log_print(*args):
            print(*args)
            print(
                *args,
                file=log_file,
            )
            log_file.flush()

        log_print(
            "SCCM drift-detection "
            "quality experiment"
        )

        log_print(
            "Started:",
            datetime.now().isoformat(
                timespec="seconds"
            ),
        )

        log_print(
            "Script:",
            script_file,
        )

        log_print(
            "Output dir:",
            output_directory,
        )

        log_print("=" * 80)

        for seed in config["seeds"]:
            try:
                seed_config = dict(config)

                seed_config[
                    "current_seed"
                ] = seed

                (
                    events,
                    meta,
                    drift_type,
                    number_of_samples,
                    processing_increment,
                    processed_samples,
                    resource_measurement,
                ) = run_one_seed(
                    seed_config,
                    seed,
                    log_file,
                )

                all_events.extend(
                    events
                )

                (
                    seed_summary,
                    drift_point_rows,
                ) = summarize_seed(
                    events,
                    seed_config,
                    meta,
                    drift_type,
                    number_of_samples,
                    processing_increment=processing_increment,
                    processed_samples=processed_samples,
                    resource_measurement=resource_measurement,
                )

                seed_summaries.append(
                    seed_summary
                )

                all_drift_point_rows.extend(
                    drift_point_rows
                )

            except Exception as error:
                trace = traceback.format_exc()

                error_rows.append({
                    "seed": seed,
                    "error": repr(error),
                    "traceback": trace,
                })

                log_print(
                    "ERROR for seed",
                    seed,
                    repr(error),
                )

                log_print(trace)

        if seed_summaries:
            final_summary = (
                aggregate_seed_summaries(
                    seed_summaries,
                    config,
                    drift_type,
                )
            )

            rows = (
                seed_summaries
                + [final_summary]
            )

            log_print(
                "\n" + "=" * 40
            )

            log_print("BRIEF RESULT")
            log_print("=" * 40)

            log_print(
                "model:",
                final_summary.get(
                    "model"
                ),
            )

            log_print(
                "dataset:",
                final_summary.get(
                    "dataset"
                ),
            )

            log_print(
                "kpi:",
                final_summary.get(
                    "kpi"
                ),
            )

            log_print(
                "multiplier:",
                final_summary.get(
                    "multiplier"
                ),
            )

            log_print(
                "sccm_window_size:",
                final_summary.get(
                    "sccm_window_size"
                ),
            )

            log_print(
                "used_kpi_window_size:",
                final_summary.get(
                    "used_kpi_window_size"
                ),
            )

            log_print(
                "increment_user_value:",
                final_summary.get(
                    "increment_user_value"
                ),
            )

            log_print(
                "TP:",
                final_summary.get("tp"),
            )

            log_print(
                "FP:",
                final_summary.get("fp"),
            )

            log_print(
                "FN:",
                final_summary.get("fn"),
            )

            log_print(
                "Precision:",
                final_summary.get(
                    "precision"
                ),
            )

            log_print(
                "Recall:",
                final_summary.get(
                    "recall"
                ),
            )

            log_print(
                "F1:",
                final_summary.get("f1"),
            )

            log_print(
                "Mean delay samples:",
                final_summary.get(
                    "mean_delay_samples"
                ),
            )

            log_print(
                "Mean delay processing increments:",
                final_summary.get(
                    "mean_delay_increments"
                ),
            )

            log_print(
                "Practical delay batches:",
                final_summary.get(
                    "practical_delay_batches"
                ),
            )

            log_print(
                "Candidate triggers:",
                final_summary.get(
                    "candidate_triggers"
                ),
            )

            log_print(
                "Raw candidate confirmations:",
                final_summary.get(
                    "raw_candidate_confirmations"
                ),
            )

            log_print(
                "Alarm episodes:",
                final_summary.get(
                    "alarm_episodes"
                ),
            )

            log_print(
                "Adaptations:",
                final_summary.get(
                    "adaptations"
                ),
            )

            log_print(
                "Recalibrations:",
                final_summary.get(
                    "recalibrations"
                ),
            )

            log_print(
                "=" * 40 + "\n"
            )

        else:
            rows = []

        write_csv(
            all_events,
            events_csv,
        )

        write_csv(
            rows,
            summary_csv,
        )

        write_csv(
            all_drift_point_rows,
            drift_points_csv,
        )

        write_csv(
            error_rows,
            errors_csv,
        )

        log_print("\nSaved:")
        log_print(" -", events_csv)
        log_print(" -", summary_csv)
        log_print(" -", drift_points_csv)
        log_print(" -", errors_csv)
        log_print(" -", console_log)

        log_print(
            "Finished:",
            datetime.now().isoformat(
                timespec="seconds"
            ),
        )

    expected_seed_count = len(config.get("seeds", DEFAULT_SEEDS))
    if error_rows or len(seed_summaries) != expected_seed_count:
        raise RuntimeError(
            f"Incomplete SCCM experiment for {config['model']}/{config['dataset']}: "
            f"completed {len(seed_summaries)} of {expected_seed_count} seeds; "
            f"errors={len(error_rows)}. See {errors_csv}."
        )

    return {
        "events_csv":
            str(events_csv),
        "summary_csv":
            str(summary_csv),
        "drift_points_csv":
            str(drift_points_csv),
        "errors_csv":
            str(errors_csv),
        "console_log":
            str(console_log),
    }


def collect_dataset_aggregate_rows(
    base_dir=None,
):
    base_dir = Path(
        base_dir or DDQ_ROOT
    )

    rows = []

    for summary_file in base_dir.rglob(
        "quality_outputs/*_summary.csv"
    ):
        for row in read_csv(summary_file):
            if (
                row.get("row_type")
                == "aggregate_for_dataset"
            ):
                row["source_file"] = str(
                    summary_file.relative_to(
                        base_dir
                    )
                )

                rows.append(row)

    return rows


def to_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return default

        return float(value)

    except Exception:
        return default


def aggregate_rows_for_paper(
    dataset_rows,
):
    grouped = {}

    for row in dataset_rows:
        key = (
            row.get("model"),
            row.get("drift_type"),
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    output = []

    for (
        model,
        drift_type,
    ), rows in sorted(
        grouped.items()
    ):
        true_drifts = sum(
            int(
                to_float(
                    row.get(
                        "true_drifts"
                    )
                )
            )
            for row in rows
        )

        candidate_triggers = sum(
            int(
                to_float(
                    row.get(
                        "candidate_triggers"
                    )
                )
            )
            for row in rows
        )

        raw_candidate_confirmations = sum(
            int(
                to_float(
                    row.get(
                        "raw_candidate_confirmations"
                    )
                )
            )
            for row in rows
        )

        raw_alarm_episodes = sum(
            int(
                to_float(
                    row.get(
                        "raw_alarm_episodes"
                    )
                )
            )
            for row in rows
        )

        alarm_episodes = sum(
            int(
                to_float(
                    row.get(
                        "alarm_episodes"
                    )
                )
            )
            for row in rows
        )

        duplicate_triggers = sum(
            int(
                to_float(
                    row.get(
                        "duplicate_candidate_triggers"
                    )
                )
            )
            for row in rows
        )

        tp = sum(
            int(
                to_float(
                    row.get("tp")
                )
            )
            for row in rows
        )

        fp = sum(
            int(
                to_float(
                    row.get("fp")
                )
            )
            for row in rows
        )

        fn = sum(
            int(
                to_float(
                    row.get("fn")
                )
            )
            for row in rows
        )

        adaptations = sum(
            int(
                to_float(
                    row.get(
                        "adaptations"
                    )
                )
            )
            for row in rows
        )

        recalibrations = sum(
            int(
                to_float(
                    row.get(
                        "recalibrations"
                    )
                )
            )
            for row in rows
        )

        recalibration_batches = sum(
            int(
                to_float(
                    row.get(
                        "recalibration_batches"
                    )
                )
            )
            for row in rows
        )

        delay_sum = 0.0
        delay_count = 0

        for row in rows:
            mean_delay = row.get(
                "mean_delay",
                "",
            )

            row_tp = int(
                to_float(
                    row.get("tp")
                )
            )

            if (
                mean_delay != ""
                and row_tp > 0
            ):
                delay_sum += (
                    float(mean_delay)
                    * row_tp
                )

                delay_count += row_tp

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        mean_delay = (
            delay_sum / delay_count
            if delay_count
            else ""
        )

        output.append({
            "model": model,
            "drift_type": drift_type,
            "dataset_count": len(rows),
            "true_drifts":
                true_drifts,
            "candidate_triggers":
                candidate_triggers,
            "raw_candidate_confirmations":
                raw_candidate_confirmations,
            "raw_alarm_episodes":
                raw_alarm_episodes,
            "alarm_episodes":
                alarm_episodes,
            "duplicate_candidate_triggers":
                duplicate_triggers,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(
                precision,
                4,
            ),
            "recall": round(
                recall,
                4,
            ),
            "f1": round(
                f1,
                4,
            ),
            "mean_delay": (
                round(
                    mean_delay,
                    2,
                )
                if mean_delay != ""
                else ""
            ),
            "adaptations":
                adaptations,
            "recalibrations":
                recalibrations,
            "recalibration_batches":
                recalibration_batches,
        })

    return output