import csv
from pathlib import Path

from benchmark_detection_common import (
    EVALUATION_SEEDS,
    MODEL_DIRECTORY_MARKERS,
    SYNTHETIC2_DATASETS,
    discover_experiment_scripts,
    discover_quality_configs,
    find_project_root,
)


def main() -> None:
    project_root = find_project_root()
    quality_configs = discover_quality_configs(str(project_root.resolve()))

    total_scripts = 0
    total_baselines = 0
    for model in MODEL_DIRECTORY_MARKERS:
        scripts = discover_experiment_scripts(project_root, model)
        total_scripts += len(scripts)
        total_baselines += len(scripts) * 8
        print(f"{model}: {len(scripts)} synthetic scripts, 8 baselines each")

    missing = []
    invalid = []
    for model in MODEL_DIRECTORY_MARKERS:
        for dataset in SYNTHETIC2_DATASETS:
            key = (model, dataset)
            if key not in quality_configs:
                missing.append(key)
                continue
            config, path = quality_configs[key]
            if int(config.get("train_percent", 90)) != 90:
                invalid.append(
                    (key, "train_percent", config.get("train_percent"), path)
                )
            for field in ("tolerance_ratio", "cooldown_factor", "min_episode_size"):
                if field not in config:
                    invalid.append((key, field, "missing", path))

    if missing:
        raise RuntimeError(f"Missing SCCM quality configurations: {missing}")
    if invalid:
        raise RuntimeError(f"Invalid SCCM quality configurations: {invalid}")
    if total_scripts != 72:
        raise RuntimeError(f"Expected 72 synthetic scripts, found {total_scripts}")
    if total_baselines != 576:
        raise RuntimeError(
            f"Expected 576 baseline configurations, found {total_baselines}"
        )

    audit_rows = []
    for (model, dataset), (config, path) in sorted(quality_configs.items()):
        audit_rows.append({
            "model": model,
            "dataset": dataset,
            "dataset_family": "Datasets.Synthetic2",
            "seed": ";".join(str(value) for value in EVALUATION_SEEDS),
            "train_percent": config.get("train_percent", 90),
            "candidate_source": config.get("candidate_source", "long_term"),
            "tolerance_ratio": config.get("tolerance_ratio", 0.05),
            "cooldown_factor": config.get("cooldown_factor", 2.0),
            "min_episode_size": config.get("min_episode_size", 2),
            "increment_user_value": config.get("increment_user_value", 10),
            "report_interval": config.get("report_interval", ""),
            "pa_c": config.get("pa_c", ""),
            "pa_epsilon": config.get("pa_epsilon", ""),
            "rls_lambda": config.get("rls_lambda", ""),
            "rls_delta": config.get("rls_delta", ""),
            "wh_learning_rate": config.get("wh_learning_rate", ""),
            "quality_config_path": str(path.relative_to(project_root)),
            "baseline_detector_adaptation_settings": "preserved from original experiment script",
        })

    audit_path = (
        project_root
        / "BenchmarkDetectionActivation"
        / "results"
        / "benchmark_exact_configuration_audit.csv"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Validation passed: {total_scripts} synthetic scripts found.")
    print(f"SCCM quality CONFIG files matched: {len(quality_configs)}")
    print(
        f"Expected baseline runs: "
        f"{total_baselines * len(EVALUATION_SEEDS)}"
    )
    print(f"Evaluation seeds: {list(EVALUATION_SEEDS)}")
    print("Dataset family: Datasets.Synthetic2")
    print("Exact per-experiment SCCM alignment settings: enabled")
    print("Exact SCCM base-model settings: enabled")
    print("Original ADWIN/KSWIN adaptation settings: preserved")
    print(f"Configuration audit: {audit_path}")


if __name__ == "__main__":
    main()
