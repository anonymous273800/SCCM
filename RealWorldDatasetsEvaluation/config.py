from __future__ import annotations

SEEDS = [0, 1, 42, 123, 7]
DATASETS = ["CCPP", "MCPD", "KCHSD", "1KC", "UCIAQD", "GASD", "CalCOFI", "WSSF"]
MODELS = ["OLR-WA", "PA", "RLS", "WidrowHoff"]
METHODS = [
    "BASE",
    "SCCM",
    "ADWIN-RESET",
    "ADWIN-WINDOW",
    "ADWIN-SSPT",
    "ADWIN-OHL",
    "KSWIN-RESET",
    "KSWIN-WINDOW",
    "KSWIN-SSPT",
    "KSWIN-OHL",
]

EXPECTED_RUNS = len(SEEDS) * len(DATASETS) * len(MODELS) * len(METHODS)

DATASET_SETTINGS = {
    "CCPP":   {"train_percent": 90, "increment_size": 50,  "report_interval": 1},
    "MCPD":   {"train_percent": 90, "increment_size": 50,  "report_interval": 1},
    "KCHSD":  {"train_percent": 90, "increment_size": 50,  "report_interval": 1},
    "1KC":    {"train_percent": 90, "increment_size": 50,  "report_interval": 1},
    "UCIAQD": {"train_percent": 10, "increment_size": 500, "report_interval": 500},
    "GASD":   {"train_percent": None, "increment_size": 500, "report_interval": 500},
    "CalCOFI":{"train_percent": 90, "increment_size": 500, "report_interval": 500},
    "WSSF":   {"train_percent": 90, "increment_size": 500, "report_interval": 500},
}

# These values reproduce the settings already used in the manuscript's real-data scripts.
MODEL_SETTINGS = {
    "OLR-WA": {
        "w_base": 0.5,
        "w_inc": 0.5,
        "default_base_model_size": 1.0,
        "sccm_multiplier": 1.5,
        "adwin_delta": 0.002,
        "kswin_alpha": 0.005,
        "kswin_window_size": 100,
        "kswin_stat_size": 30,
        "window_size_in_batches": 5,
        "sspt_candidates": (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
        "ohl_eta": 0.1,
        "ohl_eps": 0.05,
        "bounds": (0.05, 0.95),
    },
    "PA": {
        "c": 1.0,
        "epsilon": 0.1,
        "sccm_multiplier": 1.5,
        "adwin_delta": 0.002,
        "kswin_alpha": 0.005,
        "kswin_window_size": 100,
        "kswin_stat_size": 30,
        "window_size": 50,
        "sspt_candidates": (0.1,0.2,0.5,1.0,2.0,5.0),
        "ohl_eta": 0.1,
        "ohl_eps": 0.05,
        "bounds": (0.05, 10.0),
    },
    "RLS": {
        "lambda": 0.99,
        "delta": 1.0,
        "sccm_multiplier": 1.5,
        "adwin_delta": 0.002,
        "kswin_alpha": 0.005,
        "kswin_window_size": 100,
        "kswin_stat_size": 30,
        "window_size": 50,
        "sspt_candidates": (0.90,0.93,0.95,0.97,0.99,0.995),
        "ohl_eta": 0.1,
        "ohl_eps": 0.01,
        "bounds": (0.85, 0.999),
    },
    "WidrowHoff": {
        "learning_rate": 0.01,
        "sccm_multiplier": 1.5,
        "adwin_delta": 0.002,
        "kswin_alpha": 0.005,
        "kswin_window_size": 100,
        "kswin_stat_size": 30,
        "window_size": 50,
        "sspt_candidates": (0.001,0.003,0.005,0.008,0.01,0.015,0.02,0.03),
        "ohl_eta": 0.02,
        "ohl_eps": 0.01,
        "bounds": (1e-4, 0.05),
    },
}
