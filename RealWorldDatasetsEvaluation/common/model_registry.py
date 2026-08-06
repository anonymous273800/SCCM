from __future__ import annotations
from typing import Callable, Any
import copy
import numpy as np

from RealWorldDatasetsEvaluation.config import MODEL_SETTINGS
from RealWorldDatasetsEvaluation.common.project import ensure_project_importable


def _rls_settings(dataset: str) -> dict:
    cfg = copy.deepcopy(MODEL_SETTINGS["RLS"])
    if dataset == "GASD":
        # GASD has 134 dimensions and a short initial base batch. Forgetting
        # factors far below one make the covariance recursion numerically
        # unstable over the full stream. These values are fixed before the
        # final seeds and shared by every RLS-based method on GASD.
        cfg.update({
            "lambda": 0.999,
            "delta": 0.1,
            "sspt_candidates": (0.995, 0.997, 0.998, 0.999, 0.9995),
            "ohl_eta": 1e-4,
            "bounds": (0.995, 0.9999),
        })
    elif dataset in {"CalCOFI", "WSSF"}:
        # Very long/high-dimensional streams need a near-unity forgetting
        # factor to prevent covariance growth. The same configuration is used
        # across BASE, SCCM, ADWIN, and KSWIN variants for fairness.
        cfg.update({
            "lambda": 1.0,
            "delta": 0.1,
            "sspt_candidates": (0.9999, 0.99999, 0.999999, 1.0),
            "ohl_eta": 1e-6,
            "bounds": (0.99999, 1.0),
        })
    return cfg


def _widrowhoff_settings(dataset: str, X: np.ndarray) -> dict:
    cfg = copy.deepcopy(MODEL_SETTINGS["WidrowHoff"])
    if dataset == "GASD":
        # Bound standardized GASD features and choose a conservative LMS step
        # size from the resulting worst-case squared norm. The same fixed rate
        # and search range are used by all Widrow-Hoff variants.
        clip_value = 5.0
        max_norm_sq = 1.0 + float(X.shape[1]) * clip_value * clip_value
        safe_lr = float(min(cfg["learning_rate"], 0.05 / max_norm_sq))
        cfg.update({
            "learning_rate": safe_lr,
            "sspt_candidates": tuple(safe_lr * f for f in (0.25, 0.5, 0.75, 1.0)),
            "ohl_eta": safe_lr * 0.05,
            "ohl_eps": 0.01,
            "bounds": (safe_lr * 0.1, safe_lr),
            "feature_clip": (-clip_value, clip_value),
        })
    return cfg


def build_call(model: str, method: str, dataset: str, data) -> tuple[Callable[[], Any], dict]:
    ensure_project_importable()
    X, y, X_test, y_test = data.fit_X, data.fit_y, data.test_X, data.test_y
    ri = data.report_interval

    if model == "OLR-WA":
        from Models.OLR_WA import OLR_WA, OLR_WA_SCCM
        from Models.OLR_WA import OLR_WA_ADWIN_RESET, OLR_WA_ADWIN_WINDOW, OLR_WA_ADWIN_SSPT, OLR_WA_ADWIN_OHL
        from Models.OLR_WA import OLR_WA_KSWIN_RESET, OLR_WA_KSWIN_WINDOW, OLR_WA_KSWIN_SSPT, OLR_WA_KSWIN_OHL
        c = MODEL_SETTINGS[model]
        common = (X, y, c["w_base"], c["w_inc"], data.olr_base_model_size, data.increment_size, X_test, y_test)
        calls = {
            "BASE": lambda: OLR_WA.olr_wa(*common),
            "SCCM": lambda: OLR_WA_SCCM.olr_wa_sccm(*common, kpi="R2", multiplier=c["sccm_multiplier"]),
            "ADWIN-RESET": lambda: OLR_WA_ADWIN_RESET.olr_wa_regression_adversarial_adwin_reset(*common, adwin_delta=c["adwin_delta"]),
            "ADWIN-WINDOW": lambda: OLR_WA_ADWIN_WINDOW.olr_wa_regression_adversarial_adwin_window(*common, adwin_delta=c["adwin_delta"], window_size_in_batches=c["window_size_in_batches"]),
            "ADWIN-SSPT": lambda: OLR_WA_ADWIN_SSPT.olr_wa_regression_adversarial_adwin_sspt(*common, adwin_delta=c["adwin_delta"], sspt_w_inc_candidates=c["sspt_candidates"], sspt_metric="r2"),
            "ADWIN-OHL": lambda: OLR_WA_ADWIN_OHL.olr_wa_regression_adversarial_adwin_ohl(*common, adwin_delta=c["adwin_delta"], ohl_eta=c["ohl_eta"], ohl_eps=c["ohl_eps"], w_inc_bounds=c["bounds"]),
            "KSWIN-RESET": lambda: OLR_WA_KSWIN_RESET.olr_wa_regression_adversarial_kswin_reset(*common, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"]),
            "KSWIN-WINDOW": lambda: OLR_WA_KSWIN_WINDOW.olr_wa_regression_adversarial_kswin_window(*common, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], window_size_in_batches=c["window_size_in_batches"]),
            "KSWIN-SSPT": lambda: OLR_WA_KSWIN_SSPT.olr_wa_regression_adversarial_kswin_sspt(*common, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], sspt_w_inc_candidates=c["sspt_candidates"], sspt_metric="r2"),
            "KSWIN-OHL": lambda: OLR_WA_KSWIN_OHL.olr_wa_regression_adversarial_kswin_ohl(*common, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], ohl_eta=c["ohl_eta"], ohl_eps=c["ohl_eps"], w_inc_bounds=c["bounds"]),
        }
        return calls[method], c

    if model == "PA":
        from Models.PA import PA, PA_SCCM
        from Models.PA import PA_ADWIN_RESET, PA_ADWIN_WINDOW, PA_ADWIN_SSPT, PA_ADWIN_OHL
        from Models.PA import PA_KSWIN_RESET, PA_KSWIN_WINDOW, PA_KSWIN_SSPT, PA_KSWIN_OHL
        c = MODEL_SETTINGS[model]
        base = (X, y, c["c"], c["epsilon"], X_test, y_test)
        calls = {
            "BASE": lambda: PA.pa_generic(*base, report_interval=ri),
            "SCCM": lambda: PA_SCCM.ad_pa_generic(*base, kpi="MSE", multiplier=c["sccm_multiplier"], report_interval=ri, ds=dataset, c_bounds=c["bounds"]),
            "ADWIN-RESET": lambda: PA_ADWIN_RESET.pa_generic_adwin_reset(*base, adwin_delta=c["adwin_delta"], reset_mode="window", window_size=c["window_size"], report_interval=ri),
            "ADWIN-WINDOW": lambda: PA_ADWIN_WINDOW.pa_generic_adwin_window(*base, adwin_delta=c["adwin_delta"], window_size=c["window_size"], report_interval=ri),
            "ADWIN-SSPT": lambda: PA_ADWIN_SSPT.pa_generic_adwin_sspt(*base, adwin_delta=c["adwin_delta"], sspt_c_candidates=c["sspt_candidates"], window_size=c["window_size"], report_interval=ri),
            "ADWIN-OHL": lambda: PA_ADWIN_OHL.pa_generic_adwin_ohl(*base, adwin_delta=c["adwin_delta"], ohl_eta=c["ohl_eta"], ohl_eps=c["ohl_eps"], c_bounds=c["bounds"], report_interval=ri),
            "KSWIN-RESET": lambda: PA_KSWIN_RESET.pa_generic_kswin_reset(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], reset_mode="window", window_size=c["window_size"], report_interval=ri),
            "KSWIN-WINDOW": lambda: PA_KSWIN_WINDOW.pa_generic_kswin_window(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], window_size=c["window_size"], report_interval=ri),
            "KSWIN-SSPT": lambda: PA_KSWIN_SSPT.pa_generic_kswin_sspt(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], sspt_c_candidates=c["sspt_candidates"], window_size=c["window_size"], report_interval=ri),
            "KSWIN-OHL": lambda: PA_KSWIN_OHL.pa_generic_kswin_ohl(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], ohl_eta=c["ohl_eta"], ohl_eps=c["ohl_eps"], c_bounds=c["bounds"], report_interval=ri),
        }
        return calls[method], c

    if model == "RLS":
        from Models.RLS import RLS, RLS_SCCM
        from Models.RLS import RLS_ADWIN_RESET, RLS_ADWIN_WINDOW, RLS_ADWIN_SSPT, RLS_ADWIN_OHL
        from Models.RLS import RLS_KSWIN_RESET, RLS_KSWIN_WINDOW, RLS_KSWIN_SSPT, RLS_KSWIN_OHL
        c = _rls_settings(dataset)
        base = (X, y, c["lambda"], c["delta"], X_test, y_test)
        calls = {
            "BASE": lambda: RLS.rls_generic(*base, report_interval=ri),
            "SCCM": lambda: RLS_SCCM.ad_rls_generic(*base, kpi="MSE", multiplier=c["sccm_multiplier"], DS=dataset, report_interval=ri, lambda_bounds=c["bounds"]),
            "ADWIN-RESET": lambda: RLS_ADWIN_RESET.rls_generic_adwin_reset(*base, adwin_delta=c["adwin_delta"], window_size=c["window_size"], report_interval=ri),
            "ADWIN-WINDOW": lambda: RLS_ADWIN_WINDOW.rls_generic_adwin_window(*base, adwin_delta=c["adwin_delta"], window_size=c["window_size"], report_interval=ri),
            "ADWIN-SSPT": lambda: RLS_ADWIN_SSPT.rls_generic_adwin_sspt(*base, adwin_delta=c["adwin_delta"], sspt_lambda_candidates=c["sspt_candidates"], window_size=c["window_size"], report_interval=ri),
            "ADWIN-OHL": lambda: RLS_ADWIN_OHL.rls_generic_adwin_ohl(*base, adwin_delta=c["adwin_delta"], ohl_eta=c["ohl_eta"], ohl_eps=c["ohl_eps"], lambda_bounds=c["bounds"], report_interval=ri),
            "KSWIN-RESET": lambda: RLS_KSWIN_RESET.rls_generic_kswin_reset(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], window_size=c["window_size"], report_interval=ri),
            "KSWIN-WINDOW": lambda: RLS_KSWIN_WINDOW.rls_generic_kswin_window(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], window_size=c["window_size"], report_interval=ri),
            "KSWIN-SSPT": lambda: RLS_KSWIN_SSPT.rls_generic_kswin_sspt(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], sspt_lambda_candidates=c["sspt_candidates"], window_size=c["window_size"], report_interval=ri),
            "KSWIN-OHL": lambda: RLS_KSWIN_OHL.rls_generic_kswin_ohl(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], ohl_eta=c["ohl_eta"], ohl_eps=c["ohl_eps"], lambda_bounds=c["bounds"], report_interval=ri),
        }
        return calls[method], c

    if model == "WidrowHoff":
        from Models.WidrowHoff import WidrowHoff, WidrowHoff_SCCM
        from Models.WidrowHoff import WidrowHoff_ADWIN_RESET, WidrowHoff_ADWIN_WINDOW, WidrowHoff_ADWIN_SSPT, WidrowHoff_ADWIN_OHL
        from Models.WidrowHoff import WidrowHoff_KSWIN_RESET, WidrowHoff_KSWIN_WINDOW, WidrowHoff_KSWIN_SSPT, WidrowHoff_KSWIN_OHL
        c = _widrowhoff_settings(dataset, X)
        if "feature_clip" in c:
            low, high = c["feature_clip"]
            X = np.clip(np.asarray(X, dtype=float), low, high)
            X_test = np.clip(np.asarray(X_test, dtype=float), low, high)
        base = (X, y, c["learning_rate"], X_test, y_test)
        calls = {
            "BASE": lambda: WidrowHoff.widrow_hoff_generic(*base, report_interval=ri),
            "SCCM": lambda: WidrowHoff_SCCM.ad_widrow_hoff_generic(
                *base, kpi="MSE", multiplier=c["sccm_multiplier"], DS=dataset,
                report_interval=ri, lr_bounds=c.get("bounds")
            ),
            "ADWIN-RESET": lambda: WidrowHoff_ADWIN_RESET.widrow_hoff_generic_adwin_reset(*base, adwin_delta=c["adwin_delta"], window_size=c["window_size"], report_interval=ri),
            "ADWIN-WINDOW": lambda: WidrowHoff_ADWIN_WINDOW.widrow_hoff_generic_adwin_window(*base, adwin_delta=c["adwin_delta"], window_size=c["window_size"], report_interval=ri),
            "ADWIN-SSPT": lambda: WidrowHoff_ADWIN_SSPT.widrow_hoff_generic_adwin_sspt(*base, adwin_delta=c["adwin_delta"], sspt_lr_candidates=c["sspt_candidates"], window_size=c["window_size"], report_interval=ri),
            "ADWIN-OHL": lambda: WidrowHoff_ADWIN_OHL.widrow_hoff_generic_adwin_ohl(*base, adwin_delta=c["adwin_delta"], ohl_eta=c["ohl_eta"], ohl_eps=c["ohl_eps"], lr_bounds=c["bounds"], report_interval=ri),
            "KSWIN-RESET": lambda: WidrowHoff_KSWIN_RESET.widrow_hoff_generic_kswin_reset(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], window_size=c["window_size"], report_interval=ri),
            "KSWIN-WINDOW": lambda: WidrowHoff_KSWIN_WINDOW.widrow_hoff_generic_kswin_window(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], window_size=c["window_size"], report_interval=ri),
            "KSWIN-SSPT": lambda: WidrowHoff_KSWIN_SSPT.widrow_hoff_generic_kswin_sspt(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], sspt_lr_candidates=c["sspt_candidates"], window_size=c["window_size"], report_interval=ri),
            "KSWIN-OHL": lambda: WidrowHoff_KSWIN_OHL.widrow_hoff_generic_kswin_ohl(*base, kswin_alpha=c["kswin_alpha"], kswin_window_size=c["kswin_window_size"], kswin_stat_size=c["kswin_stat_size"], ohl_eta=c["ohl_eta"], ohl_eps=c["ohl_eps"], lr_bounds=c["bounds"], report_interval=ri),
        }
        return calls[method], c

    raise KeyError(f"Unknown model: {model}")
