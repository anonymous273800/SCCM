from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import csv
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Iterable


DIRECTORY_NAME = "BenchmarkDetectionActivation"
RESULTS_DIRECTORY_NAME = "results"
EVALUATION_SEEDS = (0, 1, 42, 123, 7)
METADATA_SEED = EVALUATION_SEEDS[0]

MODEL_DIRECTORY_MARKERS = {
    "OLR-WA": "001-OLR-WA",
    "PA": "002-PA",
    "RLS": "003-RLS",
    "WidrowHoff": "004-WidrowHoff",
}

BASE_MODEL_ALIASES = {"OLR_WA", "PA", "RLS", "WidrowHoff"}
ADAPTATIONS = ("RESET", "WINDOW", "SSPT", "OHL")

SYNTHETIC2_DATASETS = {
    "ADS01": ("Datasets.Synthetic2.Abrupt.ADS01", "get_DS01", "abrupt"),
    "ADS02": ("Datasets.Synthetic2.Abrupt.ADS02", "get_DS02", "abrupt"),
    "ADS03": ("Datasets.Synthetic2.Abrupt.ADS03", "get_DS03", "abrupt"),
    "ADS04": ("Datasets.Synthetic2.Abrupt.ADS04", "get_DS04", "abrupt"),
    "ADS05": ("Datasets.Synthetic2.Abrupt.ADS05", "get_DS05", "abrupt"),
    "ADS06": ("Datasets.Synthetic2.Abrupt.ADS06", "get_DS06", "abrupt"),
    "IDS01": ("Datasets.Synthetic2.Incremental.IDS01", "get_IDS01", "incremental"),
    "IDS02": ("Datasets.Synthetic2.Incremental.IDS02", "get_IDS02", "incremental"),
    "IDS03": ("Datasets.Synthetic2.Incremental.IDS03", "get_IDS03", "incremental"),
    "IDS04": ("Datasets.Synthetic2.Incremental.IDS04", "get_IDS04", "incremental"),
    "IDS05": ("Datasets.Synthetic2.Incremental.IDS05", "get_IDS05", "incremental"),
    "IDS06": ("Datasets.Synthetic2.Incremental.IDS06", "get_IDS06", "incremental"),
    "GDS01": ("Datasets.Synthetic2.Gradual.GDS01", "get_GDS01", "gradual"),
    "GDS02": ("Datasets.Synthetic2.Gradual.GDS02", "get_GDS02", "gradual"),
    "GDS03": ("Datasets.Synthetic2.Gradual.GDS03", "get_GDS03", "gradual"),
    "GDS04": ("Datasets.Synthetic2.Gradual.GDS04", "get_GDS04", "gradual"),
    "GDS05": ("Datasets.Synthetic2.Gradual.GDS05", "get_GDS05", "gradual"),
    "GDS06": ("Datasets.Synthetic2.Gradual.GDS06", "get_GDS06", "gradual"),
}

DETECTION_RE = re.compile(
    r"^(ADWIN|KSWIN) detected drift at global sample index:\s*(-?\d+)\s*$"
)
TOTAL_RE = re.compile(r"^Total\s+.+?:\s*(\d+)\s*$")


@dataclass
class ExperimentMetadata:
    drift_type: str = ""
    dataset: str = ""
    source_script: str = ""
    train_percent: int = 90
    full_dataset_samples: int = 0
    monitored_samples: int = 0
    all_true_drift_points: list[int] = field(default_factory=list)
    observable_true_drift_points: list[int] = field(default_factory=list)
    quality_config_path: str = ""
    quality_config_json: str = ""
    candidate_source: str = "long_term"
    tolerance_ratio: float = 0.05
    cooldown_factor: float = 2.0
    min_episode_size: int = 2
    increment_user_value: int = 10


@dataclass
class ActivityRecord:
    drift_type: str
    dataset: str
    model: str
    seed: int
    baseline: str
    detector: str
    adaptation: str
    source_script: str
    configuration_json: str
    train_percent: int
    full_dataset_samples: int
    monitored_samples: int
    all_true_drift_points: list[int]
    observable_true_drift_points: list[int]
    quality_config_path: str
    quality_config_json: str
    candidate_source: str
    tolerance_ratio: float
    cooldown_factor: float
    min_episode_size: int
    increment_user_value: int
    detector_detections: int = 0
    adaptation_activations: int = 0
    reported_activation_total: str = ""
    detection_indices: list[int] = field(default_factory=list)
    status: str = "ok"
    error: str = ""

    def as_row(self) -> dict[str, Any]:
        if self.reported_activation_total == "":
            count_consistent: str | bool = ""
        else:
            count_consistent = (
                int(self.reported_activation_total) == self.adaptation_activations
            )

        return {
            "drift_type": self.drift_type,
            "dataset": self.dataset,
            "model": self.model,
            "seed": self.seed,
            "baseline": self.baseline,
            "detector": self.detector,
            "adaptation": self.adaptation,
            "train_percent": self.train_percent,
            "full_dataset_samples": self.full_dataset_samples,
            "monitored_samples": self.monitored_samples,
            "all_true_drift_points": ";".join(map(str, self.all_true_drift_points)),
            "observable_true_drift_points": ";".join(
                map(str, self.observable_true_drift_points)
            ),
            "quality_config_path": self.quality_config_path,
            "quality_config_json": self.quality_config_json,
            "candidate_source": self.candidate_source,
            "tolerance_ratio": self.tolerance_ratio,
            "cooldown_factor": self.cooldown_factor,
            "min_episode_size": self.min_episode_size,
            "increment_user_value": self.increment_user_value,
            "detector_detections": self.detector_detections,
            "adaptation_activations": self.adaptation_activations,
            "detections_minus_activations": (
                self.detector_detections - self.adaptation_activations
            ),
            "reported_activation_total": self.reported_activation_total,
            "activation_count_consistent": count_consistent,
            "detection_indices": ";".join(map(str, self.detection_indices)),
            "configuration_json": self.configuration_json,
            "source_script": self.source_script,
            "status": self.status,
            "error": self.error,
        }


class ActivityPrintCollector:
    """Suppress verbose model output while counting existing log markers."""

    def __init__(self, record: ActivityRecord):
        self.record = record

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(arg) for arg in args)
        if end is not None:
            text += str(end)

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            detection_match = DETECTION_RE.match(line)
            if detection_match:
                detector_name = detection_match.group(1)
                if detector_name == self.record.detector:
                    self.record.detector_detections += 1
                    self.record.detection_indices.append(
                        int(detection_match.group(2))
                    )
                continue

            if self._is_activation_line(line):
                self.record.adaptation_activations += 1
                continue

            total_match = TOTAL_RE.match(line)
            if total_match and self._is_relevant_total_line(line):
                self.record.reported_activation_total = total_match.group(1)

    def _is_activation_line(self, line: str) -> bool:
        adaptation = self.record.adaptation

        if adaptation == "RESET":
            return line == "RESET ACTIVATED"
        if adaptation == "WINDOW":
            return line == "WINDOW RETRAIN ACTIVATED"
        if adaptation == "SSPT":
            return line == "SSPT ACTIVATED" or (
                self.record.model != "OLR-WA" and line.startswith("SSPT tuned ")
            )
        if adaptation == "OHL":
            return line == "OHL ACTIVATED" or (
                self.record.model != "OLR-WA" and line.startswith("OHL tuned ")
            )
        return False

    def _is_relevant_total_line(self, line: str) -> bool:
        lower = line.lower()
        detector_present = self.record.detector.lower() in lower
        if not detector_present:
            return False

        if self.record.adaptation == "RESET":
            return "reset" in lower
        if self.record.adaptation == "WINDOW":
            return "window retrain" in lower
        if self.record.adaptation == "SSPT":
            return "sspt" in lower
        if self.record.adaptation == "OHL":
            return "ohl" in lower
        return False


class ActivityRegistry:
    def __init__(self, project_root: Path, model: str):
        self.project_root = project_root
        self.model = model
        self.records: list[ActivityRecord] = []
        self.failures: list[dict[str, str]] = []
        self.metadata = ExperimentMetadata()
        self.current_seed = METADATA_SEED

    def set_experiment(self, metadata: ExperimentMetadata) -> None:
        self.metadata = metadata

    def add_failure(self, stage: str, error: BaseException) -> None:
        self.failures.append(
            {
                "model": self.model,
                "drift_type": self.metadata.drift_type,
                "dataset": self.metadata.dataset,
                "source_script": self.metadata.source_script,
                "stage": stage,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )

    def create_record(
        self,
        *,
        baseline: str,
        detector: str,
        adaptation: str,
        configuration_json: str,
    ) -> ActivityRecord:
        metadata = self.metadata
        return ActivityRecord(
            drift_type=metadata.drift_type,
            dataset=metadata.dataset,
            model=self.model,
            seed=self.current_seed,
            baseline=baseline,
            detector=detector,
            adaptation=adaptation,
            source_script=metadata.source_script,
            configuration_json=configuration_json,
            train_percent=metadata.train_percent,
            full_dataset_samples=metadata.full_dataset_samples,
            monitored_samples=metadata.monitored_samples,
            all_true_drift_points=list(metadata.all_true_drift_points),
            observable_true_drift_points=list(
                metadata.observable_true_drift_points
            ),
            quality_config_path=metadata.quality_config_path,
            quality_config_json=metadata.quality_config_json,
            candidate_source=metadata.candidate_source,
            tolerance_ratio=metadata.tolerance_ratio,
            cooldown_factor=metadata.cooldown_factor,
            min_episode_size=metadata.min_episode_size,
            increment_user_value=metadata.increment_user_value,
        )


@contextlib.contextmanager
def patched_print(new_print: Callable[..., None]):
    original_print = builtins.print
    builtins.print = new_print
    try:
        yield
    finally:
        builtins.print = original_print


@contextlib.contextmanager
def patched_attributes(patches: Iterable[tuple[Any, str, Any]]):
    originals: list[tuple[Any, str, Any]] = []
    try:
        for owner, attribute_name, replacement in patches:
            originals.append((owner, attribute_name, getattr(owner, attribute_name)))
            setattr(owner, attribute_name, replacement)
        yield
    finally:
        for owner, attribute_name, original in reversed(originals):
            setattr(owner, attribute_name, original)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    while current != current.parent:
        if (
            (current / "Experiments" / "001 Synthetic").is_dir()
            and (current / "Models").is_dir()
            and (current / "Datasets").is_dir()
        ):
            return current
        current = current.parent

    raise RuntimeError("Could not locate the SCCM project root.")


def _has_experiment_functions(script_path: Path) -> bool:
    try:
        tree = ast.parse(script_path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return False

    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {
        "run_single_seed_experiment",
        "run_multi_seed_experiment",
    }.issubset(function_names)


def discover_experiment_scripts(project_root: Path, model: str) -> list[Path]:
    marker = MODEL_DIRECTORY_MARKERS[model]
    synthetic_root = project_root / "Experiments" / "001 Synthetic"

    scripts: list[Path] = []
    for script_path in synthetic_root.rglob("*.py"):
        lower_name = script_path.name.lower()
        if marker not in str(script_path):
            continue
        if any(word in lower_name for word in ("aggregate", "visualized", "delme")):
            continue
        if _has_experiment_functions(script_path):
            scripts.append(script_path)

    scripts = sorted(scripts)
    if len(scripts) != 18:
        raise RuntimeError(
            f"Expected 18 synthetic scripts for {model}, found {len(scripts)}."
        )
    return scripts


def import_experiment_module(script_path: Path):
    module_hash = hashlib.sha1(str(script_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"benchmark_detection_experiment_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import experiment script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_main_guard(tree: ast.Module) -> ast.If | None:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test_text = ast.dump(node.test, include_attributes=False)
        if "__name__" in test_text and "__main__" in test_text:
            return node
    return None


def _extract_run_multi_call(statement: ast.stmt) -> ast.Call | None:
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_multi_seed_experiment"
        ):
            return node
    return None


def _evaluate_ast_expression(
    expression: ast.AST | None,
    environment: dict[str, Any],
    script_path: Path,
) -> Any:
    if expression is None:
        return None
    compiled = compile(ast.Expression(expression), filename=str(script_path), mode="eval")
    return eval(compiled, environment, environment)


def extract_main_run_kwargs(script_path: Path, module: Any) -> dict[str, Any]:
    tree = ast.parse(script_path.read_text(encoding="utf-8", errors="ignore"))
    main_guard = _find_main_guard(tree)
    if main_guard is None:
        raise RuntimeError(f"No __main__ block found in {script_path}")

    environment = dict(module.__dict__)
    for statement in main_guard.body:
        call = _extract_run_multi_call(statement)
        if call is not None:
            kwargs: dict[str, Any] = {}
            for keyword in call.keywords:
                if keyword.arg is None:
                    raise RuntimeError("**kwargs is not supported in the original call.")
                kwargs[keyword.arg] = _evaluate_ast_expression(
                    keyword.value, environment, script_path
                )
            return kwargs

        if isinstance(statement, ast.Assign):
            value = _evaluate_ast_expression(statement.value, environment, script_path)
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    environment[target.id] = value
                    module.__dict__[target.id] = value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            value = _evaluate_ast_expression(statement.value, environment, script_path)
            environment[statement.target.id] = value
            module.__dict__[statement.target.id] = value

    raise RuntimeError(f"Could not find run_multi_seed_experiment call in {script_path}")


def extract_called_module_functions(
    script_path: Path, function_name: str
) -> list[tuple[str, str]]:
    tree = ast.parse(script_path.read_text(encoding="utf-8", errors="ignore"))
    function_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    if function_node is None:
        raise RuntimeError(f"Function {function_name} not found in {script_path}")

    calls: list[tuple[str, str]] = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if isinstance(node.func.value, ast.Name):
            pair = (node.func.value.id, node.func.attr)
            if pair not in calls:
                calls.append(pair)
    return calls


def extract_call_return_arities(
    script_path: Path, function_name: str
) -> dict[tuple[str, str], int]:
    tree = ast.parse(script_path.read_text(encoding="utf-8", errors="ignore"))
    function_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    if function_node is None:
        return {}

    arities: dict[tuple[str, str], int] = {}
    for statement in function_node.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if value is None:
            continue

        target = statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
        arity = len(target.elts) if isinstance(target, (ast.Tuple, ast.List)) else 1

        for node in ast.walk(value):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if isinstance(node.func.value, ast.Name):
                arities[(node.func.value.id, node.func.attr)] = arity
    return arities


def infer_drift_type(script_path: Path, main_kwargs: dict[str, Any]) -> str:
    value = main_kwargs.get("DRIFT_TYPE")
    if value is not None:
        return str(value).strip().lower()

    path_text = str(script_path).lower()
    for drift_type in ("abrupt", "incremental", "gradual"):
        if drift_type in path_text:
            return drift_type
    raise RuntimeError(f"Could not infer drift type from {script_path}")


def _extract_literal_assignment(script_path: Path, variable_name: str) -> Any:
    tree = ast.parse(script_path.read_text(encoding="utf-8", errors="ignore"))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable_name for target in targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{variable_name} not found in {script_path}")


@lru_cache(maxsize=1)
def discover_quality_configs(project_root_text: str) -> dict[tuple[str, str], tuple[dict[str, Any], Path]]:
    project_root = Path(project_root_text)
    quality_root = project_root / "DriftDetectionQuality"
    mapping: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for path in quality_root.rglob("quality_run.py"):
        config = dict(_extract_literal_assignment(path, "CONFIG"))
        key = (str(config["model"]), str(config["dataset"]).upper())
        if key in mapping:
            raise RuntimeError(f"Duplicate quality CONFIG for {key}: {path}")
        mapping[key] = (config, path)
    if len(mapping) != 72:
        raise RuntimeError(f"Expected 72 SCCM quality CONFIG files, found {len(mapping)}")
    return mapping


def get_quality_config(project_root: Path, model: str, dataset: str) -> tuple[dict[str, Any], Path]:
    mapping = discover_quality_configs(str(project_root.resolve()))
    key = (model, dataset.upper())
    if key not in mapping:
        raise RuntimeError(f"No SCCM quality CONFIG found for {key}")
    config, path = mapping[key]
    return dict(config), path


def get_synthetic2_module(dataset: str):
    module_name, _, _ = SYNTHETIC2_DATASETS[dataset.upper()]
    return importlib.import_module(module_name)


def get_synthetic2_getter(dataset: str) -> Callable[..., Any]:
    module_name, getter_name, _ = SYNTHETIC2_DATASETS[dataset.upper()]
    module = importlib.import_module(module_name)
    return getattr(module, getter_name)


def _find_dataset_module_alias(script_path: Path) -> str:
    tree = ast.parse(script_path.read_text(encoding="utf-8", errors="ignore"))
    function_node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_multi_seed_experiment"),
        None,
    )
    if function_node is None:
        raise RuntimeError("run_multi_seed_experiment not found")
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr.startswith(("get_", "print_")) and isinstance(node.func.value, ast.Name):
            return node.func.value.id
    raise RuntimeError(f"Could not identify dataset module alias in {script_path}")


def module_model_name(script_path: Path) -> str:
    path_text = str(script_path)
    for model, marker in MODEL_DIRECTORY_MARKERS.items():
        if marker in path_text:
            return model
    raise RuntimeError(f"Could not infer model from {script_path}")


def _find_dataset_getter(script_path: Path, module: Any) -> Callable[..., Any]:
    tree = ast.parse(script_path.read_text(encoding="utf-8", errors="ignore"))
    function_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_multi_seed_experiment"
        ),
        None,
    )
    if function_node is None:
        raise RuntimeError("run_multi_seed_experiment not found")

    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not node.func.attr.startswith("get_"):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        owner = getattr(module, node.func.value.id, None)
        if owner is not None and hasattr(owner, node.func.attr):
            return getattr(owner, node.func.attr)

    raise RuntimeError(f"Could not identify dataset getter in {script_path}")


def true_drift_points_from_meta(meta: dict[str, Any], drift_type: str) -> list[int]:
    if drift_type == "abrupt":
        return [int(meta["drift_point"])]

    if drift_type == "incremental":
        step = int(meta["samples_per_step"])
        n_steps = int(meta["n_steps"])
        return [step * index for index in range(1, n_steps)]

    if drift_type == "gradual":
        segment_lengths = [int(value) for value in meta["segment_lengths"]]
        points: list[int] = []
        running = 0
        for length in segment_lengths[:-1]:
            running += length
            points.append(running)
        return points

    raise ValueError(f"Unsupported drift type: {drift_type}")


def build_experiment_metadata(
    project_root: Path,
    module: Any,
    script_path: Path,
    main_kwargs: dict[str, Any],
) -> ExperimentMetadata:
    drift_type = infer_drift_type(script_path, main_kwargs)
    dataset = str(main_kwargs.get("DATASET_NAME", script_path.stem)).strip().upper()
    train_percent = int(main_kwargs.get("TRAIN_PERCENT", 90))
    model = module_model_name(script_path)

    quality_config, quality_path = get_quality_config(project_root, model, dataset)
    quality_train_percent = int(quality_config.get("train_percent", 90))
    if quality_train_percent != train_percent:
        raise RuntimeError(
            f"Train-percent mismatch for {model}/{dataset}: "
            f"experiment={train_percent}, quality={quality_train_percent}"
        )

    getter = get_synthetic2_getter(dataset)
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        output = getter(seed=METADATA_SEED, return_meta=True)

    X = output[0]
    meta = output[-1]
    full_dataset_samples = int(len(X))
    monitored_samples = int(train_percent * full_dataset_samples / 100)
    all_points = true_drift_points_from_meta(meta, drift_type)
    observable_points = [point for point in all_points if point < monitored_samples]

    return ExperimentMetadata(
        drift_type=drift_type,
        dataset=dataset,
        source_script=str(script_path.relative_to(project_root)),
        train_percent=train_percent,
        full_dataset_samples=full_dataset_samples,
        monitored_samples=monitored_samples,
        all_true_drift_points=all_points,
        observable_true_drift_points=observable_points,
        quality_config_path=str(quality_path.relative_to(project_root)),
        quality_config_json=json.dumps(quality_config, sort_keys=True, default=str),
        candidate_source=str(quality_config.get("candidate_source", "long_term")),
        tolerance_ratio=float(quality_config.get("tolerance_ratio", 0.05)),
        cooldown_factor=float(quality_config.get("cooldown_factor", 2.0)),
        min_episode_size=int(quality_config.get("min_episode_size", 2)),
        increment_user_value=int(quality_config.get("increment_user_value", 10)),
    )


def build_configuration_json(
    function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    try:
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
        values = bound.arguments
    except (TypeError, ValueError):
        values = kwargs

    excluded_names = {
        "X", "y", "X_train", "y_train", "X_test", "y_test",
        "recent_X", "recent_y",
    }
    configuration: dict[str, Any] = {}
    for name, value in values.items():
        if name in excluded_names or name.lower().startswith(("x_", "y_")):
            continue
        safe_value = _to_json_safe(value)
        if safe_value is not None:
            configuration[name] = safe_value

    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        converted = [_to_json_safe(item) for item in value]
        if all(item is not None for item in converted):
            return converted
    return None


def make_baseline_wrapper(
    *,
    original_function: Callable[..., Any],
    registry: ActivityRegistry,
    detector: str,
    adaptation: str,
    return_arity: int,
) -> Callable[..., Any]:
    baseline_name = f"{registry.model}-{detector}-{adaptation}"

    def wrapped(*args: Any, **kwargs: Any):
        record = registry.create_record(
            baseline=baseline_name,
            detector=detector,
            adaptation=adaptation,
            configuration_json=build_configuration_json(
                original_function, args, kwargs
            ),
        )
        collector = ActivityPrintCollector(record)

        try:
            with patched_print(collector):
                result = original_function(*args, **kwargs)
        except Exception as error:
            record.status = "error"
            record.error = f"{type(error).__name__}: {error}"
            values: list[Any] = [0.0]
            values.extend([] for _ in range(max(0, return_arity - 1)))
            result = tuple(values)

        registry.records.append(record)
        return result

    wrapped.__name__ = getattr(original_function, "__name__", "wrapped_baseline")
    wrapped.__doc__ = getattr(original_function, "__doc__", None)
    return wrapped


def make_dummy_model_function(
    original_function: Callable[..., Any], return_arity: int
) -> Callable[..., Any]:
    def dummy(*args: Any, **kwargs: Any):
        values: list[Any] = [0.0]
        values.extend([] for _ in range(max(0, return_arity - 1)))
        return tuple(values)

    dummy.__name__ = getattr(original_function, "__name__", "dummy_model")
    return dummy


def make_run_single_wrapper(
    original_function: Callable[..., Any], registry: ActivityRegistry
) -> Callable[..., Any]:
    signature = inspect.signature(original_function)

    def wrapped(*args: Any, **kwargs: Any):
        try:
            bound = signature.bind_partial(*args, **kwargs)
            registry.current_seed = int(bound.arguments.get("seed", METADATA_SEED))
        except (TypeError, ValueError):
            registry.current_seed = int(kwargs.get("seed", METADATA_SEED))

        result = original_function(*args, **kwargs)
        _fill_empty_metric_lists(result)
        return result

    return wrapped


def _fill_empty_metric_lists(result: Any) -> None:
    if not isinstance(result, dict):
        return

    metric_dicts = [value for value in result.values() if isinstance(value, dict)]
    template_r2: list[float] | None = None
    template_mse: list[float] | None = None

    for metrics in metric_dicts:
        r2_values = metrics.get("R2")
        mse_values = metrics.get("MSE")
        if template_r2 is None and _sequence_length(r2_values) > 0:
            template_r2 = [0.0] * _sequence_length(r2_values)
        if template_mse is None and _sequence_length(mse_values) > 0:
            template_mse = [0.0] * _sequence_length(mse_values)

    template_r2 = template_r2 or [0.0]
    template_mse = template_mse or [0.0]

    for metrics in metric_dicts:
        if "R2" in metrics and _sequence_length(metrics["R2"]) == 0:
            metrics["R2"] = list(template_r2)
        if "MSE" in metrics and _sequence_length(metrics["MSE"]) == 0:
            metrics["MSE"] = list(template_mse)


def _sequence_length(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def build_consistency_patches(
    module: Any,
    script_path: Path,
    metadata: ExperimentMetadata,
) -> list[tuple[Any, str, Any]]:
    quality_config = json.loads(metadata.quality_config_json)
    patches: list[tuple[Any, str, Any]] = []

    # Replace the original Synthetic dataset module with the exact Synthetic2
    # module used by the SCCM drift-quality experiment.
    dataset_alias = _find_dataset_module_alias(script_path)
    patches.append((module, dataset_alias, get_synthetic2_module(metadata.dataset)))

    def patch_module_value(name: str, value: Any) -> None:
        if hasattr(module, name):
            patches.append((module, name, value))

    model = module_model_name(script_path)
    if model == "OLR-WA":
        increment_value = int(quality_config.get("increment_user_value", 10))
        hyperparameter_module = getattr(module, "Hyperparameter", None)
        if hyperparameter_module is None or not hasattr(hyperparameter_module, "olr_wa_increment_size"):
            raise RuntimeError(
                f"Hyperparameter.olr_wa_increment_size not found in {script_path}"
            )
        patches.append(
            (
                hyperparameter_module,
                "olr_wa_increment_size",
                lambda number_of_features, user_defined_val=increment_value: increment_value,
            )
        )
    elif model == "PA":
        patch_module_value("PA_C", float(quality_config["pa_c"]))
        patch_module_value("PA_EPSILON", float(quality_config["pa_epsilon"]))
        patch_module_value("PA_C_BOUNDS", tuple(quality_config["pa_c_bounds"]))
        patch_module_value("REPORT_INTERVAL", int(quality_config.get("report_interval", 10)))
    elif model == "RLS":
        patch_module_value("RLS_LAMBDA", float(quality_config["rls_lambda"]))
        patch_module_value("RLS_DELTA", float(quality_config["rls_delta"]))
        patch_module_value("RLS_LAMBDA_BOUNDS", tuple(quality_config["rls_lambda_bounds"]))
        patch_module_value("REPORT_INTERVAL", int(quality_config.get("report_interval", 10)))
    elif model == "WidrowHoff":
        patch_module_value("WH_LEARNING_RATE", float(quality_config["wh_learning_rate"]))
        patch_module_value("REPORT_INTERVAL", int(quality_config.get("report_interval", 1)))

    return patches


def build_experiment_patches(
    module: Any,
    script_path: Path,
    registry: ActivityRegistry,
) -> list[tuple[Any, str, Any]]:
    patches: list[tuple[Any, str, Any]] = []
    calls = extract_called_module_functions(script_path, "run_single_seed_experiment")
    return_arities = extract_call_return_arities(
        script_path, "run_single_seed_experiment"
    )

    baseline_count = 0
    for module_alias, function_name in calls:
        owner = getattr(module, module_alias, None)
        if owner is None or not hasattr(owner, function_name):
            continue

        original_function = getattr(owner, function_name)
        alias_upper = module_alias.upper()
        return_arity = return_arities.get((module_alias, function_name), 3)

        if "ADWIN" in alias_upper or "KSWIN" in alias_upper:
            detector = "ADWIN" if "ADWIN" in alias_upper else "KSWIN"
            adaptation = next(
                name for name in ADAPTATIONS if name in alias_upper
            )
            replacement = make_baseline_wrapper(
                original_function=original_function,
                registry=registry,
                detector=detector,
                adaptation=adaptation,
                return_arity=return_arity,
            )
            patches.append((owner, function_name, replacement))
            baseline_count += 1
            continue

        if "SCCM" in alias_upper or module_alias in BASE_MODEL_ALIASES:
            patches.append(
                (
                    owner,
                    function_name,
                    make_dummy_model_function(original_function, return_arity),
                )
            )

    if baseline_count != 8:
        raise RuntimeError(
            f"Expected 8 baseline calls in {script_path}, found {baseline_count}."
        )

    patches.append(
        (
            module,
            "run_single_seed_experiment",
            make_run_single_wrapper(module.run_single_seed_experiment, registry),
        )
    )

    quantify_drift = getattr(module, "QuantifyDrift", None)
    if quantify_drift is not None and hasattr(
        quantify_drift, "save_drift_metrics_to_excel"
    ):
        patches.append(
            (
                quantify_drift,
                "save_drift_metrics_to_excel",
                lambda *args, **kwargs: None,
            )
        )

    return patches


def run_experiment_script(
    script_path: Path,
    registry: ActivityRegistry,
    runtime_root: Path,
) -> None:
    module = import_experiment_module(script_path)
    main_kwargs = extract_main_run_kwargs(script_path, module)
    main_kwargs["seeds"] = list(EVALUATION_SEEDS)
    if "PLOTTING_ENABLED" in main_kwargs:
        main_kwargs["PLOTTING_ENABLED"] = False

    metadata = build_experiment_metadata(
        registry.project_root, module, script_path, main_kwargs
    )
    registry.set_experiment(metadata)

    run_directory = runtime_root / registry.model / metadata.drift_type / metadata.dataset
    run_directory.mkdir(parents=True, exist_ok=True)
    if "PLOTTING_DIR" in main_kwargs:
        main_kwargs["PLOTTING_DIR"] = str(run_directory)

    patches = build_experiment_patches(module, script_path, registry)
    patches.extend(build_consistency_patches(module, script_path, metadata))
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with (
            patched_attributes(patches),
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            module.run_multi_seed_experiment(**main_kwargs)


def _file_safe_model(model: str) -> str:
    return model.lower().replace("-", "_")


def write_records(path: Path, records: list[ActivityRecord]) -> None:
    empty = ActivityRecord(
        drift_type="", dataset="", model="", seed=METADATA_SEED, baseline="",
        detector="", adaptation="", source_script="", configuration_json="",
        train_percent=90, full_dataset_samples=0, monitored_samples=0,
        all_true_drift_points=[], observable_true_drift_points=[],
        quality_config_path="", quality_config_json="",
        candidate_source="long_term", tolerance_ratio=0.05,
        cooldown_factor=2.0, min_episode_size=2, increment_user_value=10,
    )
    fieldnames = list(empty.as_row().keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())


def write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    fieldnames = [
        "model", "drift_type", "dataset", "source_script",
        "stage", "error", "traceback",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def run_model(model: str) -> tuple[Path, Path]:
    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    results_root = project_root / DIRECTORY_NAME / RESULTS_DIRECTORY_NAME
    raw_root = results_root / "raw"
    runtime_root = results_root / "runtime"
    raw_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    registry = ActivityRegistry(project_root, model)
    scripts = discover_experiment_scripts(project_root, model)

    print(f"Model: {model}")
    print(f"Synthetic experiment scripts: {len(scripts)}")
    print(f"Evaluation seeds: {list(EVALUATION_SEEDS)}")

    for index, script_path in enumerate(scripts, start=1):
        relative_path = script_path.relative_to(project_root)
        print(f"[{index:02d}/{len(scripts):02d}] {relative_path}")
        try:
            run_experiment_script(script_path, registry, runtime_root)
        except Exception as error:
            registry.add_failure("experiment", error)
            print(f"  ERROR: {type(error).__name__}: {error}")

        records_path = raw_root / f"benchmark_activity_{_file_safe_model(model)}.csv"
        failures_path = raw_root / f"benchmark_failures_{_file_safe_model(model)}.csv"
        write_records(records_path, registry.records)
        write_failures(failures_path, registry.failures)

    expected_rows = len(scripts) * 8 * len(EVALUATION_SEEDS)
    print(f"Completed baseline rows: {len(registry.records)} (expected {expected_rows})")
    print(f"Experiment failures: {len(registry.failures)}")
    print(f"Detailed results: {records_path}")
    print(f"Failure log: {failures_path}")
    return records_path, failures_path


def cli_main(default_model: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the synthetic ADWIN/KSWIN detector-adaptation baselines "
            "using the exact Synthetic2 data, SCCM model settings, "
            "SCCM alignment settings, and five evaluation seeds."
        )
    )
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_DIRECTORY_MARKERS.keys()],
        default=default_model or "all",
    )
    args = parser.parse_args()

    models = list(MODEL_DIRECTORY_MARKERS) if args.model == "all" else [args.model]
    for model in models:
        run_model(model)


if __name__ == "__main__":
    cli_main()
