"""Sealed-holdout-safe temporal feature and model challenger experiments."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .application_baseline import assign_splits, evaluate_probabilities
from .application_features import engineer_application_features, select_responsible_features
from .application_lightgbm import _preprocessor

HISTORY_FLAGS = (
    "HAS_BUREAU_HISTORY",
    "HAS_PREVIOUS_APPLICATION",
    "HAS_POS_HISTORY",
    "HAS_CREDIT_CARD_HISTORY",
    "HAS_INSTALLMENT_HISTORY",
)
STRUCTURAL_PRUNING = {
    "PREV_APPLICATION_COUNT",
    "BUREAU_CREDIT_COUNT",
    "CREDIT_TO_GOODS",
}


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    temporal_selector: str
    model_kind: str = "lightgbm_standard"
    drop_external_scores: bool = False
    structural_pruning: bool = False


VARIANTS = (
    ExperimentVariant("v1_plus_repayment_recency", "repayment_recency"),
    ExperimentVariant("v1_plus_repayment_sequence", "repayment_sequence"),
    ExperimentVariant("v1_plus_installment_temporal", "installment_all"),
    ExperimentVariant("v1_plus_previous_velocity", "previous_velocity"),
    ExperimentVariant("v1_plus_pos_bureau_burden", "pos_bureau_burden"),
    ExperimentVariant("v1_plus_credit_cycling", "credit_cycling"),
    ExperimentVariant("v1_plus_credit_card_temporal", "credit_card_temporal"),
    ExperimentVariant("v1_plus_all_temporal", "all"),
    ExperimentVariant(
        "v1_plus_all_structural_pruning", "all", structural_pruning=True
    ),
    ExperimentVariant(
        "v1_plus_all_without_external_scores", "all", drop_external_scores=True
    ),
    ExperimentVariant(
        "v1_plus_all_high_capacity", "all", model_kind="lightgbm_high_capacity"
    ),
    ExperimentVariant("v1_plus_all_catboost", "all", model_kind="catboost"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.astype(float).where(denominator > 0))


def engineer_temporal_cross_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add transparent application/history burden interactions for research."""

    result = frame.copy()
    income = result["AMT_INCOME_TOTAL"]
    active_annuity = result["TBUREAU_ACTIVE_ANNUITY_SUM"]
    active_debt = result["TBUREAU_ACTIVE_DEBT_SUM"]
    result["BURDEN_ACTIVE_ANNUITY_TO_INCOME"] = _safe_ratio(active_annuity, income)
    result["BURDEN_ACTIVE_DEBT_TO_INCOME"] = _safe_ratio(active_debt, income)
    result["BURDEN_PROPOSED_PLUS_ACTIVE_ANNUITY_TO_INCOME"] = _safe_ratio(
        result["AMT_ANNUITY"] + active_annuity, income
    )
    result["BURDEN_REQUESTED_PLUS_ACTIVE_DEBT_TO_INCOME"] = _safe_ratio(
        result["AMT_CREDIT"] + active_debt, income
    )
    availability = [
        column for column in result if column.startswith("HAS_TEMPORAL_")
    ]
    result["BURDEN_TEMPORAL_SOURCE_COUNT"] = result[availability].sum(axis=1)
    result["CYCLING_SIGNAL_BURDEN_INTERACTION"] = (
        result["CYCLING_SIGNAL_FAMILY_COUNT"]
        * result["BURDEN_PROPOSED_PLUS_ACTIVE_ANNUITY_TO_INCOME"]
    )
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def _is_sequence_feature(column: str) -> bool:
    return column.startswith("TINST_RECENT_EVENT_") or any(
        token in column for token in ("STREAK", "CONSECUTIVE", "CURE", "RELAPSE")
    )


def _is_recency_feature(column: str) -> bool:
    return column.startswith("TINST_") and not _is_sequence_feature(column)


def temporal_columns_for_variant(columns: list[str], selector: str) -> list[str]:
    """Return the predeclared additive columns for one ablation."""

    availability = [column for column in columns if column.startswith("HAS_TEMPORAL_")]
    if selector == "repayment_recency":
        selected = [column for column in columns if _is_recency_feature(column)]
    elif selector == "repayment_sequence":
        selected = [column for column in columns if _is_sequence_feature(column)]
    elif selector == "installment_all":
        selected = [column for column in columns if column.startswith("TINST_")]
    elif selector == "previous_velocity":
        selected = [column for column in columns if column.startswith("TPREV_")]
    elif selector == "pos_bureau_burden":
        selected = [
            column
            for column in columns
            if column.startswith(("TPOS_", "TBUREAU_", "BURDEN_"))
        ]
    elif selector == "credit_card_temporal":
        selected = [column for column in columns if column.startswith("TCC_")]
    elif selector == "credit_cycling":
        exact = {
            "TPREV_APPLICATION_BURST_MAX_30D",
            "TPREV_REFUSED_COUNT_90D",
            "TPREV_OTHER_LOAN_PURPOSE_COUNT",
            "TPREV_CASH_AFTER_STRESS_90D_COUNT",
            "TPOS_ACTIVE_CONCURRENCY_MAX_12M",
            "TCC_MAXED_RATE_6M",
            "TCC_BELOW_MINIMUM_RATE_6M",
            "BURDEN_PROPOSED_PLUS_ACTIVE_ANNUITY_TO_INCOME",
        }
        selected = [
            column for column in columns if column.startswith("CYCLING_") or column in exact
        ]
    elif selector == "all":
        selected = [
            column
            for column in columns
            if column.startswith(
                ("TINST_", "TPREV_", "TPOS_", "TCC_", "TBUREAU_", "CYCLING_", "BURDEN_")
            )
        ]
    else:
        raise ValueError(f"unknown temporal selector: {selector}")
    return sorted(set(selected + availability))


def _lightgbm_model(kind: str, n_estimators: int, random_seed: int) -> LGBMClassifier:
    parameters = {
        "objective": "binary",
        "n_estimators": n_estimators,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if kind == "lightgbm_high_capacity":
        parameters.update(
            {
                "learning_rate": 0.02,
                "num_leaves": 63,
                "min_child_samples": 60,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.20,
                "reg_lambda": 1.5,
            }
        )
    return LGBMClassifier(**parameters)


def _calibration_coefficients(target: np.ndarray, probability: np.ndarray) -> dict:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    recalibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    recalibrator.fit(logit, target)
    return {
        "calibration_intercept": float(recalibrator.intercept_[0]),
        "calibration_slope": float(recalibrator.coef_[0, 0]),
    }


def _evaluation(target: pd.Series, probability: np.ndarray) -> dict:
    metrics = evaluate_probabilities(target, probability)
    metrics.update(_calibration_coefficients(target.to_numpy(dtype=int), probability))
    return metrics


def _paired_bootstrap(
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    iterations: int,
    random_seed: int,
) -> dict:
    rng = np.random.default_rng(random_seed)
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    ap_deltas = np.empty(iterations, dtype=float)
    brier_deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        sampled_target = target[sample]
        ap_deltas[iteration] = average_precision_score(
            sampled_target, candidate[sample]
        ) - average_precision_score(sampled_target, baseline[sample])
        brier_deltas[iteration] = brier_score_loss(
            sampled_target, candidate[sample]
        ) - brier_score_loss(sampled_target, baseline[sample])
    return {
        "iterations": iterations,
        "average_precision_delta": {
            "mean": float(ap_deltas.mean()),
            "ci_95": [float(np.quantile(ap_deltas, 0.025)), float(np.quantile(ap_deltas, 0.975))],
            "probability_positive": float((ap_deltas > 0).mean()),
        },
        "brier_delta": {
            "mean": float(brier_deltas.mean()),
            "ci_95": [
                float(np.quantile(brier_deltas, 0.025)),
                float(np.quantile(brier_deltas, 0.975)),
            ],
            "probability_improved": float((brier_deltas < 0).mean()),
        },
    }


def _segment_labels(frame: pd.DataFrame) -> pd.DataFrame:
    source_count = frame[list(HISTORY_FLAGS)].fillna(0).sum(axis=1)
    labels = pd.DataFrame(index=frame.index)
    labels["history_segment"] = pd.cut(
        source_count,
        bins=[-1, 1, 3, 5],
        labels=["thin_0_to_1", "partial_2_to_3", "full_4_to_5"],
    ).astype("string")
    labels["gender"] = frame["CODE_GENDER"].astype("string")
    age_years = -frame["DAYS_BIRTH"] / 365.25
    labels["age_band"] = pd.cut(
        age_years,
        bins=[0, 30, 40, 50, 60, 200],
        labels=["under_30", "30_to_39", "40_to_49", "50_to_59", "60_plus"],
    ).astype("string")
    return labels


def _segment_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    labels: pd.DataFrame,
) -> dict:
    result = {}
    for dimension in labels:
        groups = []
        for value in labels[dimension].dropna().unique():
            mask = labels[dimension].eq(value).to_numpy()
            group_target = target[mask]
            if len(group_target) < 100 or len(np.unique(group_target)) < 2:
                continue
            group_probability = probability[mask]
            groups.append(
                {
                    "segment": str(value),
                    "rows": int(mask.sum()),
                    "events": int(group_target.sum()),
                    "prevalence": float(group_target.mean()),
                    "average_precision": float(
                        average_precision_score(group_target, group_probability)
                    ),
                    "roc_auc": float(roc_auc_score(group_target, group_probability)),
                    "brier_score": float(brier_score_loss(group_target, group_probability)),
                }
            )
        result[dimension] = groups
    return result


def _feature_family(feature: str) -> str:
    for prefix, label in (
        ("TINST_", "Temporal installments"),
        ("TPREV_", "Previous application velocity"),
        ("TPOS_", "POS obligation and trajectory"),
        ("TCC_", "Temporal revolving credit"),
        ("TBUREAU_", "Active bureau burden"),
        ("CYCLING_", "Credit-cycling proxy"),
        ("BURDEN_", "Cross-source burden"),
        ("EXT_SOURCE", "External scores"),
        ("INST_", "Frozen installment aggregates"),
        ("PREV_", "Frozen previous applications"),
        ("POS_", "Frozen POS history"),
        ("CC_", "Frozen credit-card history"),
        ("BB_", "Frozen bureau monthly status"),
        ("BUREAU_", "Frozen bureau history"),
    ):
        if feature.startswith(prefix):
            return label
    return "Application and other"


def _lightgbm_importance(bundle: dict) -> list[dict]:
    names = list(bundle["preprocessor"].get_feature_names_out())
    gains = bundle["model"].booster_.feature_importance(importance_type="gain").astype(float)
    grouped: dict[str, float] = {}
    for name, gain in zip(names, gains, strict=True):
        raw = name.split("__", 1)[-1].removeprefix("missingindicator_")
        family = _feature_family(raw)
        grouped[family] = grouped.get(family, 0.0) + float(gain)
    total = sum(grouped.values())
    return [
        {"family": family, "gain": gain, "gain_share": gain / total}
        for family, gain in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    ]


def _train_lightgbm_variant(
    frame: pd.DataFrame,
    target: pd.Series,
    development_mask: pd.Series,
    calibration_mask: pd.Series,
    *,
    variant: ExperimentVariant,
    output_dir: Path,
    random_seed: int,
) -> tuple[np.ndarray, dict, dict]:
    selection = select_responsible_features(frame.loc[development_mask])
    selected = list(selection.selected)
    numeric = list(selection.numeric)
    categorical = list(selection.categorical)
    development_indices = np.flatnonzero(development_mask.to_numpy())
    fit_indices, early_stop_indices = train_test_split(
        development_indices,
        test_size=0.15,
        random_state=random_seed,
        stratify=target.iloc[development_indices],
    )
    tuning_preprocessor = _preprocessor(numeric, categorical)
    tuning_train = tuning_preprocessor.fit_transform(frame.iloc[fit_indices][selected])
    tuning_validation = tuning_preprocessor.transform(frame.iloc[early_stop_indices][selected])
    tuning_model = _lightgbm_model(variant.model_kind, 2500, random_seed)
    tuning_model.fit(
        tuning_train,
        target.iloc[fit_indices].to_numpy(),
        eval_X=tuning_validation,
        eval_y=target.iloc[early_stop_indices].to_numpy(),
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=125, verbose=False)],
    )
    best_iteration = int(tuning_model.best_iteration_ or tuning_model.n_estimators)
    del tuning_train, tuning_validation, tuning_preprocessor, tuning_model
    gc.collect()

    preprocessor = _preprocessor(numeric, categorical)
    development_matrix = preprocessor.fit_transform(frame.loc[development_mask, selected])
    calibration_matrix = preprocessor.transform(frame.loc[calibration_mask, selected])
    model = _lightgbm_model(variant.model_kind, best_iteration, random_seed)
    model.fit(development_matrix, target.loc[development_mask].to_numpy())
    probability = model.predict_proba(calibration_matrix)[:, 1]
    bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "feature_selection": selection.to_dict(),
        "model_name": variant.name,
        "model_version": "0.3.0-research",
        "random_seed": random_seed,
        "best_iteration": best_iteration,
        "holdout_evaluated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{variant.name}.joblib"
    joblib.dump(bundle, model_path)
    details = {
        "selected_features": len(selected),
        "numeric_features": len(numeric),
        "categorical_features": len(categorical),
        "transformed_features": int(development_matrix.shape[1]),
        "best_iteration": best_iteration,
        "artifact": {"path": str(model_path), "sha256": _sha256(model_path)},
    }
    importance = {"families": _lightgbm_importance(bundle)}
    del development_matrix, calibration_matrix
    gc.collect()
    return probability, details, importance


def _prepare_catboost(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in categorical:
        result[column] = result[column].astype("string").fillna("<MISSING>").astype(str)
    return result


def _train_catboost_variant(
    frame: pd.DataFrame,
    target: pd.Series,
    development_mask: pd.Series,
    calibration_mask: pd.Series,
    *,
    variant: ExperimentVariant,
    output_dir: Path,
    random_seed: int,
) -> tuple[np.ndarray, dict, dict]:
    selection = select_responsible_features(frame.loc[development_mask])
    selected = list(selection.selected)
    categorical = list(selection.categorical)
    prepared = _prepare_catboost(frame[selected], categorical)
    development_indices = np.flatnonzero(development_mask.to_numpy())
    fit_indices, early_stop_indices = train_test_split(
        development_indices,
        test_size=0.15,
        random_state=random_seed,
        stratify=target.iloc[development_indices],
    )
    tuning = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.035,
        depth=7,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_seed=random_seed,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )
    tuning.fit(
        prepared.iloc[fit_indices],
        target.iloc[fit_indices],
        cat_features=categorical,
        eval_set=(prepared.iloc[early_stop_indices], target.iloc[early_stop_indices]),
        early_stopping_rounds=125,
        verbose=False,
    )
    best_iteration = max(1, int(tuning.get_best_iteration()) + 1)
    del tuning
    gc.collect()
    model = CatBoostClassifier(
        iterations=best_iteration,
        learning_rate=0.035,
        depth=7,
        loss_function="Logloss",
        l2_leaf_reg=5.0,
        random_seed=random_seed,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        prepared.loc[development_mask],
        target.loc[development_mask],
        cat_features=categorical,
        verbose=False,
    )
    probability = model.predict_proba(prepared.loc[calibration_mask])[:, 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{variant.name}.cbm"
    model.save_model(model_path)
    gains = model.get_feature_importance()
    grouped: dict[str, float] = {}
    for name, gain in zip(selected, gains, strict=True):
        family = _feature_family(name)
        grouped[family] = grouped.get(family, 0.0) + float(gain)
    total = sum(grouped.values())
    importance = {
        "families": [
            {"family": family, "gain": gain, "gain_share": gain / total}
            for family, gain in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
        ]
    }
    details = {
        "selected_features": len(selected),
        "numeric_features": len(selection.numeric),
        "categorical_features": len(categorical),
        "transformed_features": len(selected),
        "best_iteration": best_iteration,
        "artifact": {"path": str(model_path), "sha256": _sha256(model_path)},
    }
    del prepared, model
    gc.collect()
    return probability, details, importance


def _score_frozen_baseline(
    engineered_base: pd.DataFrame,
    calibration_mask: pd.Series,
    model_path: Path,
) -> np.ndarray:
    bundle = joblib.load(model_path)
    selected = bundle["feature_selection"]["selected"]
    matrix = bundle["preprocessor"].transform(
        engineered_base.loc[calibration_mask, selected]
    )
    return bundle["model"].predict_proba(matrix)[:, 1]


def _gates(baseline: dict, candidate: dict, uncertainty: dict, config: dict) -> dict:
    declared = config["research_promotion_gates"]
    ap_gain = candidate["average_precision"] - baseline["average_precision"]
    brier_delta = candidate["brier_score"] - baseline["brier_score"]
    roc_delta = candidate["roc_auc"] - baseline["roc_auc"]
    capture_delta = (
        candidate["top_20_percent"]["event_capture_rate"]
        - baseline["top_20_percent"]["event_capture_rate"]
    )
    checks = {
        "average_precision_gain": ap_gain >= declared["minimum_absolute_average_precision_gain"],
        "bootstrap_probability_positive": uncertainty["average_precision_delta"][
            "probability_positive"
        ]
        >= declared["minimum_probability_ap_gain_positive"],
        "brier_non_regression": brier_delta <= declared["maximum_brier_regression"],
        "roc_auc_non_regression": roc_delta >= -declared["maximum_roc_auc_regression"],
        "top_20_capture_non_regression": capture_delta
        >= -declared["maximum_top_20_capture_regression"],
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "deltas": {
            "average_precision": ap_gain,
            "brier_score": brier_delta,
            "roc_auc": roc_delta,
            "top_20_capture": capture_delta,
        },
    }


def run_temporal_challenger_experiments(
    application_path: str | Path,
    history_v1_path: str | Path,
    temporal_path: str | Path,
    baseline_model_path: str | Path,
    experiment_config_path: str | Path,
    *,
    metrics_output: str | Path,
    comparison_output: str | Path,
    predictions_output: str | Path,
    model_output_dir: str | Path,
) -> dict:
    """Run all predeclared challengers on development/calibration only."""

    config = json.loads(Path(experiment_config_path).read_text(encoding="utf-8"))
    random_seed = int(config["random_seed"])
    bootstrap_iterations = int(config["bootstrap"]["iterations"])
    application = pd.read_csv(application_path, low_memory=False)
    target_all = application["TARGET"].astype("int8")
    splits_all = assign_splits(target_all, random_seed=random_seed)
    holdout_ids = set(application.loc[splits_all.eq("holdout"), "SK_ID_CURR"].astype(int))
    allowed = ~splits_all.eq("holdout")
    application = application.loc[allowed].copy().reset_index(drop=True)
    splits = splits_all.loc[allowed].reset_index(drop=True)
    target = application["TARGET"].astype("int8")
    if set(application["SK_ID_CURR"].astype(int)) & holdout_ids:
        raise AssertionError("sealed holdout IDs entered the challenger population")

    history_v1 = pd.read_parquet(history_v1_path)
    temporal = pd.read_parquet(temporal_path)
    if history_v1["SK_ID_CURR"].duplicated().any() or temporal["SK_ID_CURR"].duplicated().any():
        raise ValueError("history feature stores must be unique by SK_ID_CURR")
    base = application.merge(history_v1, on="SK_ID_CURR", how="left", validate="one_to_one")
    engineered_base = engineer_application_features(base)
    temporal_columns = [column for column in temporal if column != "SK_ID_CURR"]
    frame = engineered_base.merge(temporal, on="SK_ID_CURR", how="left", validate="one_to_one")
    frame = engineer_temporal_cross_features(frame)
    all_added_columns = [column for column in frame if column not in engineered_base]
    development_mask = splits.eq("development")
    calibration_mask = splits.eq("calibration")
    calibration_target = target.loc[calibration_mask]
    calibration_ids = application.loc[calibration_mask, "SK_ID_CURR"].astype(int).reset_index(drop=True)
    if set(calibration_ids) & holdout_ids:
        raise AssertionError("sealed holdout IDs entered calibration output")

    baseline_probability = _score_frozen_baseline(
        engineered_base, calibration_mask, Path(baseline_model_path)
    )
    baseline_metrics = _evaluation(calibration_target, baseline_probability)
    labels = _segment_labels(base.loc[calibration_mask].reset_index(drop=True))
    results = {
        "frozen_v1_baseline": {
            "model_kind": "frozen_history_lightgbm",
            "feature_selector": "frozen_v1",
            "metrics": baseline_metrics,
            "segment_metrics": _segment_metrics(
                calibration_target.to_numpy(dtype=int), baseline_probability, labels
            ),
            "holdout_evaluated": False,
        }
    }
    prediction_frame = pd.DataFrame(
        {
            "SK_ID_CURR": calibration_ids,
            "TARGET": calibration_target.reset_index(drop=True),
            "frozen_v1_baseline": baseline_probability,
        }
    )
    output_dir = Path(model_output_dir)

    base_columns = list(engineered_base.columns)
    for variant_index, variant in enumerate(VARIANTS, start=1):
        selected_temporal = temporal_columns_for_variant(all_added_columns, variant.temporal_selector)
        variant_columns = list(dict.fromkeys(base_columns + selected_temporal))
        if variant.drop_external_scores:
            variant_columns = [
                column for column in variant_columns if not column.startswith("EXT_SOURCE")
            ]
        if variant.structural_pruning:
            variant_columns = [
                column for column in variant_columns if column not in STRUCTURAL_PRUNING
            ]
        variant_frame = frame[variant_columns]
        print(
            f"[{variant_index}/{len(VARIANTS)}] training {variant.name} "
            f"with {len(variant_columns)} available columns",
            flush=True,
        )
        if variant.model_kind == "catboost":
            probability, details, importance = _train_catboost_variant(
                variant_frame,
                target,
                development_mask,
                calibration_mask,
                variant=variant,
                output_dir=output_dir,
                random_seed=random_seed,
            )
        else:
            probability, details, importance = _train_lightgbm_variant(
                variant_frame,
                target,
                development_mask,
                calibration_mask,
                variant=variant,
                output_dir=output_dir,
                random_seed=random_seed,
            )
        metrics = _evaluation(calibration_target, probability)
        uncertainty = _paired_bootstrap(
            calibration_target.to_numpy(dtype=int),
            baseline_probability,
            probability,
            iterations=bootstrap_iterations,
            random_seed=random_seed + variant_index,
        )
        gate = _gates(baseline_metrics, metrics, uncertainty, config)
        results[variant.name] = {
            "model_kind": variant.model_kind,
            "feature_selector": variant.temporal_selector,
            "drop_external_scores": variant.drop_external_scores,
            "structural_pruning": variant.structural_pruning,
            "details": details,
            "metrics": metrics,
            "paired_uncertainty_vs_frozen_v1": uncertainty,
            "research_promotion_gate": gate,
            "segment_metrics": _segment_metrics(
                calibration_target.to_numpy(dtype=int), probability, labels
            ),
            "importance": importance,
            "holdout_evaluated": False,
        }
        prediction_frame[variant.name] = probability
        print(
            f"    AP={metrics['average_precision']:.6f} "
            f"delta={gate['deltas']['average_precision']:+.6f} "
            f"Brier={metrics['brier_score']:.6f} gate={gate['all_passed']}",
            flush=True,
        )
        del variant_frame
        gc.collect()

    standard = prediction_frame["v1_plus_all_temporal"].to_numpy()
    high_capacity = prediction_frame["v1_plus_all_high_capacity"].to_numpy()
    ensemble_probability = 0.5 * standard + 0.5 * high_capacity
    ensemble_metrics = _evaluation(calibration_target, ensemble_probability)
    ensemble_uncertainty = _paired_bootstrap(
        calibration_target.to_numpy(dtype=int),
        baseline_probability,
        ensemble_probability,
        iterations=bootstrap_iterations,
        random_seed=random_seed + 100,
    )
    results["fixed_equal_weight_lightgbm_ensemble"] = {
        "model_kind": "fixed_probability_average",
        "feature_selector": "all",
        "components": ["v1_plus_all_temporal", "v1_plus_all_high_capacity"],
        "metrics": ensemble_metrics,
        "paired_uncertainty_vs_frozen_v1": ensemble_uncertainty,
        "research_promotion_gate": _gates(
            baseline_metrics, ensemble_metrics, ensemble_uncertainty, config
        ),
        "segment_metrics": _segment_metrics(
            calibration_target.to_numpy(dtype=int), ensemble_probability, labels
        ),
        "holdout_evaluated": False,
    }
    prediction_frame["fixed_equal_weight_lightgbm_ensemble"] = ensemble_probability

    ranked = sorted(
        results,
        key=lambda name: results[name]["metrics"]["average_precision"],
        reverse=True,
    )
    passing = [
        name
        for name in ranked
        if results[name].get("research_promotion_gate", {}).get("all_passed")
    ]
    summary_rows = []
    for rank, name in enumerate(ranked, start=1):
        metrics = results[name]["metrics"]
        gate = results[name].get("research_promotion_gate", {})
        deltas = gate.get("deltas", {})
        summary_rows.append(
            {
                "rank": rank,
                "experiment": name,
                "model_kind": results[name]["model_kind"],
                "average_precision": metrics["average_precision"],
                "ap_gain_vs_frozen_v1": deltas.get("average_precision", 0.0),
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "top_10_capture": metrics["top_10_percent"]["event_capture_rate"],
                "top_20_capture": metrics["top_20_percent"]["event_capture_rate"],
                "calibration_slope": metrics["calibration_slope"],
                "promotion_gate_passed": gate.get("all_passed", False),
            }
        )
    report = {
        "run_version": "1.0.0",
        "status": "research_challenger_calibration_only",
        "experiment_config": config,
        "population": {
            "development_rows": int(development_mask.sum()),
            "calibration_rows": int(calibration_mask.sum()),
            "sealed_holdout_rows_excluded": len(holdout_ids),
            "sealed_holdout_evaluated": False,
        },
        "feature_store": {
            "v1_path": str(history_v1_path),
            "v2_path": str(temporal_path),
            "v2_columns": len(temporal_columns),
        },
        "ranked_experiments": ranked,
        "research_gate_passing_experiments": passing,
        "best_calibration_experiment": ranked[0],
        "results": results,
        "interpretation_guardrails": [
            "No experiment in this report has been evaluated on the sealed final holdout.",
            "Calibration results select only a research candidate for a new untouched evaluation.",
            "The credit-cycling block is an inferred stress proxy and does not verify EMI funding.",
        ],
    }
    metrics_destination = Path(metrics_output)
    comparison_destination = Path(comparison_output)
    predictions_destination = Path(predictions_output)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)
    comparison_destination.parent.mkdir(parents=True, exist_ok=True)
    predictions_destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(comparison_destination, index=False)
    prediction_frame.to_parquet(predictions_destination, index=False)
    return report
