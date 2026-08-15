"""Audit every frozen history-model factor on the untouched final holdout.

The audit is diagnostic only. It does not fit, calibrate, select, or tune a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from credifast.modeling.application_features import engineer_application_features

ENGINEERED_DESCRIPTIONS = {
    "EMPLOYMENT_SENTINEL": "Whether the source employment-days value used its unknown sentinel.",
    "EMPLOYMENT_YEARS": "Reported employment tenure converted from days to years.",
    "REGISTRATION_YEARS": "Years since the applicant's registration date.",
    "ID_AGE_YEARS": "Years since the applicant's identity document was changed.",
    "PHONE_CHANGE_YEARS": "Years since the applicant last changed phone details.",
    "CREDIT_TO_INCOME": "Requested credit divided by reported annual income.",
    "ANNUITY_TO_INCOME": "Proposed annual annuity divided by reported annual income.",
    "CREDIT_TO_ANNUITY": "Requested credit divided by proposed annual annuity; a term proxy.",
    "CREDIT_TO_GOODS": "Requested credit divided by goods price.",
    "DOWN_PAYMENT_RATE": "Implied down payment divided by goods price.",
    "EXT_SOURCE_MEAN": "Mean of the available external credit-score signals.",
    "EXT_SOURCE_MIN": "Minimum of the available external credit-score signals.",
    "EXT_SOURCE_MAX": "Maximum of the available external credit-score signals.",
    "EXT_SOURCE_STD": "Dispersion among the available external credit-score signals.",
    "EXT_SOURCE_COUNT": "Number of non-missing external credit-score signals.",
    "APPLICATION_MISSING_COUNT": "Count of missing application and joined-history fields.",
}

SOURCE_LABELS = {
    "BB": "Bureau monthly status",
    "BUREAU": "Credit bureau",
    "CC": "Credit card history",
    "INST": "Installment repayments",
    "POS": "POS/cash loan history",
    "PREV": "Previous applications",
}


def source_group(feature: str) -> str:
    for prefix, label in SOURCE_LABELS.items():
        if feature.startswith(prefix + "_"):
            return label
    if feature.startswith("EXT_SOURCE"):
        return "External scores"
    if feature.startswith("HAS_"):
        return "Source availability"
    if feature in ENGINEERED_DESCRIPTIONS:
        return "Application derived"
    if feature.startswith("FLAG_DOCUMENT"):
        return "Application documents"
    if any(
        feature.startswith(prefix)
        for prefix in (
            "APARTMENTS_",
            "BASEMENTAREA_",
            "COMMONAREA_",
            "ELEVATORS_",
            "ENTRANCES_",
            "FLOORSMAX_",
            "FLOORSMIN_",
            "LANDAREA_",
            "LIVINGAPARTMENTS_",
            "LIVINGAREA_",
            "NONLIVINGAPARTMENTS_",
            "NONLIVINGAREA_",
            "YEARS_BEGINEXPLUATATION_",
            "YEARS_BUILD_",
        )
    ) or feature in {
        "TOTALAREA_MODE",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    }:
        return "Application property"
    if feature.startswith("AMT_REQ_CREDIT_BUREAU"):
        return "Application credit enquiries"
    if feature.startswith("AMT_") or feature in {
        "OWN_CAR_AGE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
    }:
        return "Application financial"
    if feature.startswith("NAME_"):
        return "Application categorical"
    return "Application verified flags"


def description_for(feature: str, source_descriptions: dict[str, str]) -> str:
    if feature in ENGINEERED_DESCRIPTIONS:
        return ENGINEERED_DESCRIPTIONS[feature]
    if feature in source_descriptions:
        return source_descriptions[feature]
    group = source_group(feature)
    words = feature
    for prefix in SOURCE_LABELS:
        words = words.removeprefix(prefix + "_")
    words = words.replace("_", " ").lower()
    return f"{group} aggregate: {words}."


def intuitive_effect(feature: str, categorical: bool) -> tuple[str, str]:
    name = feature.upper()
    if categorical:
        return "category-dependent", "Risk varies by category; no numeric ordering is assumed."
    if name == "EXT_SOURCE_STD":
        return (
            "context-dependent",
            "Greater disagreement among external scores can indicate uncertainty, but has no universal risk sign.",
        )
    adverse = (
        "DPD",
        "DELINQUENT",
        "DEFAULT",
        "OVERDUE",
        "BAD_DEBT",
        "SHORTFALL",
        "SHORT_PAYMENT",
        "LATE_",
        "REFUS",
        "HIGH_YIELD",
        "DEBT_TO_CREDIT",
        "UTILIZATION",
        "CREDIT_TO_INCOME",
        "ANNUITY_TO_INCOME",
        "APPLICATION_MISSING_COUNT",
    )
    favorable = (
        "EXT_SOURCE",
        "AMT_INCOME_TOTAL",
        "EMPLOYMENT_YEARS",
        "DOWN_PAYMENT_RATE",
        "PAYMENT_RATIO",
        "APPROVAL_RATE",
    )
    if any(token in name for token in adverse):
        return "higher -> more risk", "More arrears, burden, utilization, refusals, or missingness should raise risk."
    if any(token in name for token in favorable):
        return "higher -> less risk", "Stronger score, resources, tenure, down payment, or repayment completeness should reduce risk."
    if name.startswith("HAS_") or name.endswith(("_COUNT", "_SUM")):
        return "context-dependent", "A larger count can mean deeper history or greater credit exposure; composition and recency matter."
    if name.endswith(("_MAX", "_MEAN", "_MIN", "_RATE")):
        return "context-dependent", "The aggregate can be beneficial or adverse depending on the underlying behavior and interactions."
    return "context-dependent", "No defensible monotonic sign follows from the field alone."


def transformed_to_raw(name: str) -> tuple[str, bool]:
    normalized = name.split("__", 1)[-1]
    marker = "missingindicator_"
    if normalized.startswith(marker):
        return normalized.removeprefix(marker), True
    return normalized, False


def direction_from_effect(correlation: float, delta: float) -> str:
    if np.isfinite(correlation) and abs(correlation) >= 0.15:
        return "higher -> more risk" if correlation > 0 else "higher -> less risk"
    if np.isfinite(delta) and abs(delta) >= 0.03:
        suffix = "more risk" if delta > 0 else "less risk"
        return f"non-monotonic; upper values net {suffix}"
    return "mixed / weak marginal direction"


def cramer_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left.fillna("<MISSING>"), right.fillna("<MISSING>"))
    if min(table.shape) < 2:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False)[0]
    observations = table.to_numpy().sum()
    denominator = observations * (min(table.shape) - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator else float("nan")


def substitution_label(association: float) -> str:
    if not np.isfinite(association) or association < 0.50:
        return "low measured redundancy; not a demonstrated substitute"
    if association >= 0.90:
        return "high redundancy candidate; validate with grouped ablation before removing"
    if association >= 0.75:
        return "partial substitute candidate; important interactions may still differ"
    return "related signal, but too weak to treat as a substitute"


def category_effect(series: pd.Series, shap: np.ndarray, target: np.ndarray) -> dict:
    work = pd.DataFrame(
        {
            "category": series.astype("string").fillna("<MISSING>"),
            "shap": shap,
            "target": target,
        }
    )
    summary = work.groupby("category", observed=True).agg(
        rows=("target", "size"),
        mean_shap=("shap", "mean"),
        event_rate=("target", "mean"),
    )
    eligible = summary[summary["rows"] >= max(50, int(len(work) * 0.002))]
    if eligible.empty:
        return {}
    risky = eligible.sort_values("mean_shap", ascending=False).iloc[0]
    favorable = eligible.sort_values("mean_shap").iloc[0]
    return {
        "highest_risk_category": str(eligible["mean_shap"].idxmax()),
        "highest_risk_mean_shap": float(risky["mean_shap"]),
        "highest_risk_event_rate": float(risky["event_rate"]),
        "lowest_risk_category": str(eligible["mean_shap"].idxmin()),
        "lowest_risk_mean_shap": float(favorable["mean_shap"]),
        "lowest_risk_event_rate": float(favorable["event_rate"]),
        "eligible_categories": len(eligible),
    }


def numeric_effect(
    raw: pd.Series,
    transformed: np.ndarray,
    shap: np.ndarray,
    target: np.ndarray,
) -> dict:
    valid = raw.notna().to_numpy()
    if valid.sum() < 100 or pd.Series(transformed[valid]).nunique() < 2:
        return {
            "value_shap_spearman": float("nan"),
            "value_target_spearman": float("nan"),
            "upper_minus_lower_shap": float("nan"),
            "upper_minus_lower_event_rate_pp": float("nan"),
        }
    values = pd.Series(transformed[valid])
    shap_values = pd.Series(shap[valid])
    targets = pd.Series(target[valid])
    unique = values.nunique()
    if unique == 2:
        lower = values == values.min()
        upper = values == values.max()
    else:
        q1, q3 = values.quantile([0.25, 0.75])
        lower = values <= q1
        upper = values >= q3
    shap_correlation = (
        float(values.corr(shap_values, method="spearman"))
        if shap_values.nunique() > 1
        else float("nan")
    )
    target_correlation = (
        float(values.corr(targets, method="spearman"))
        if targets.nunique() > 1
        else float("nan")
    )
    return {
        "value_shap_spearman": shap_correlation,
        "value_target_spearman": target_correlation,
        "upper_minus_lower_shap": float(shap_values[upper].mean() - shap_values[lower].mean()),
        "upper_minus_lower_event_rate_pp": float(
            100 * (targets[upper].mean() - targets[lower].mean())
        ),
    }


def load_source_descriptions(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, encoding="latin1")
    return (
        frame.dropna(subset=["Row", "Description"])
        .drop_duplicates("Row")
        .set_index("Row")["Description"]
        .astype(str)
        .to_dict()
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(column) for column in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", type=Path, default=Path("data/raw/application_train.csv"))
    parser.add_argument("--history", type=Path, default=Path("data/processed/history_features.parquet"))
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/processed/final_holdout_predictions.parquet")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/models/history_lightgbm.joblib")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/factor_audit"))
    parser.add_argument("--correlation-sample", type=int, default=25_000)
    args = parser.parse_args()

    predictions = pd.read_parquet(args.predictions, columns=["SK_ID_CURR", "TARGET"])
    application = pd.read_csv(args.application, low_memory=False)
    application = application.merge(predictions, on=["SK_ID_CURR", "TARGET"], how="inner")
    history = pd.read_parquet(args.history)
    frame = application.merge(history, on="SK_ID_CURR", how="left", validate="one_to_one")
    engineered = engineer_application_features(frame)
    target = engineered["TARGET"].to_numpy(dtype=np.int8)

    bundle = joblib.load(args.model)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    selection = bundle["feature_selection"]
    selected = selection["selected"]
    numeric = selection["numeric"]
    categorical = selection["categorical"]
    matrix = preprocessor.transform(engineered[selected])
    transformed_names = list(preprocessor.get_feature_names_out())
    contributions = model.booster_.predict(matrix, pred_contrib=True)[:, :-1]
    gains = model.booster_.feature_importance(importance_type="gain").astype(float)
    splits = model.booster_.feature_importance(importance_type="split").astype(int)
    mean_abs_shap = np.abs(contributions).mean(axis=0)
    direct_indices: dict[str, int] = {}
    indicator_indices: dict[str, int] = {}
    for index, name in enumerate(transformed_names):
        raw, is_indicator = transformed_to_raw(name)
        (indicator_indices if is_indicator else direct_indices)[raw] = index

    source_descriptions = load_source_descriptions(
        args.application.parent / "HomeCredit_columns_description.csv"
    )
    rows = []
    for feature in selected:
        direct_index = direct_indices[feature]
        indicator_index = indicator_indices.get(feature)
        categorical_feature = feature in categorical
        direct_shap = contributions[:, direct_index]
        if categorical_feature:
            effect = {
                "value_shap_spearman": float("nan"),
                "value_target_spearman": float("nan"),
                "upper_minus_lower_shap": float("nan"),
                "upper_minus_lower_event_rate_pp": float("nan"),
            }
            category_detail = category_effect(engineered[feature], direct_shap, target)
            observed_direction = "category-dependent"
        else:
            effect = numeric_effect(
                engineered[feature], matrix[:, direct_index], direct_shap, target
            )
            category_detail = {}
            observed_direction = direction_from_effect(
                effect["value_shap_spearman"], effect["upper_minus_lower_shap"]
            )
        intuitive_direction, intuition = intuitive_effect(feature, categorical_feature)
        indicator_gain = float(gains[indicator_index]) if indicator_index is not None else 0.0
        indicator_shap = (
            float(mean_abs_shap[indicator_index]) if indicator_index is not None else 0.0
        )
        missing_effect = float("nan")
        if indicator_index is not None:
            missing = matrix[:, indicator_index] > 0
            if missing.any() and (~missing).any():
                missing_effect = float(
                    contributions[missing, indicator_index].mean()
                    - contributions[~missing, indicator_index].mean()
                )
        rows.append(
            {
                "feature": feature,
                "source_group": source_group(feature),
                "data_type": "categorical" if categorical_feature else "numeric",
                "definition": description_for(feature, source_descriptions),
                "model_use": (
                    "most-frequent imputation + ordinal encoding + LightGBM splits"
                    if categorical_feature
                    else "median imputation + missingness indicator when needed + LightGBM splits"
                ),
                "intuitive_risk_direction": intuitive_direction,
                "intuitive_reason": intuition,
                "observed_model_direction": observed_direction,
                "direct_mean_abs_shap_log_odds": float(mean_abs_shap[direct_index]),
                "missing_indicator_mean_abs_shap_log_odds": indicator_shap,
                "total_mean_abs_shap_log_odds": float(mean_abs_shap[direct_index] + indicator_shap),
                "direct_gain": float(gains[direct_index]),
                "missing_indicator_gain": indicator_gain,
                "total_gain": float(gains[direct_index] + indicator_gain),
                "split_count": int(splits[direct_index] + (splits[indicator_index] if indicator_index is not None else 0)),
                "missing_rate": float(engineered[feature].isna().mean()),
                "missing_effect_log_odds": missing_effect,
                **effect,
                "category_effect_detail": json.dumps(category_detail, sort_keys=True),
            }
        )

    audit = pd.DataFrame(rows)
    audit["shap_share"] = audit["total_mean_abs_shap_log_odds"] / audit[
        "total_mean_abs_shap_log_odds"
    ].sum()
    audit["gain_share"] = audit["total_gain"] / audit["total_gain"].sum()
    importance_order = audit.sort_values("total_mean_abs_shap_log_odds", ascending=False).index
    cumulative = audit.loc[importance_order, "shap_share"].cumsum()
    audit.loc[importance_order, "importance_tier"] = np.select(
        [cumulative <= 0.50, cumulative <= 0.80, cumulative <= 0.95],
        ["Tier 1 - dominant", "Tier 2 - material", "Tier 3 - supporting"],
        default="Tier 4 - residual",
    )

    sample = engineered.sample(
        n=min(args.correlation_sample, len(engineered)), random_state=42
    )
    correlations = sample[numeric].corr(method="spearman", min_periods=500)
    associations: dict[str, list[tuple[str, float, str]]] = {name: [] for name in selected}
    pair_rows = []
    for left_index, left in enumerate(numeric):
        for right in numeric[left_index + 1 :]:
            value = correlations.at[left, right]
            if np.isfinite(value) and abs(value) >= 0.50:
                pair_rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "association": float(value),
                        "absolute_association": float(abs(value)),
                        "method": "Spearman rho on observed values",
                    }
                )
                associations[left].append((right, float(abs(value)), "Spearman rho"))
                associations[right].append((left, float(abs(value)), "Spearman rho"))
    for left_index, left in enumerate(categorical):
        for right in categorical[left_index + 1 :]:
            value = cramer_v(sample[left], sample[right])
            if np.isfinite(value) and value >= 0.50:
                pair_rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "association": value,
                        "absolute_association": value,
                        "method": "Cramer's V",
                    }
                )
                associations[left].append((right, value, "Cramer's V"))
                associations[right].append((left, value, "Cramer's V"))

    for index, row in audit.iterrows():
        related = sorted(associations[row["feature"]], key=lambda item: item[1], reverse=True)
        top = related[0] if related else ("", float("nan"), "")
        audit.loc[index, "strongest_related_feature"] = top[0]
        audit.loc[index, "strongest_absolute_association"] = top[1]
        audit.loc[index, "association_method"] = top[2]
        audit.loc[index, "related_features_ge_0_50"] = json.dumps(
            [
                {"feature": name, "absolute_association": round(value, 6), "method": method}
                for name, value, method in related[:5]
            ]
        )
        audit.loc[index, "substitutability_assessment"] = substitution_label(top[1])

    audit = audit.sort_values("total_mean_abs_shap_log_odds", ascending=False).reset_index(drop=True)
    audit.insert(0, "importance_rank", np.arange(1, len(audit) + 1))
    group_summary = (
        audit.groupby("source_group", as_index=False)
        .agg(
            factor_count=("feature", "size"),
            shap_share=("shap_share", "sum"),
            gain_share=("gain_share", "sum"),
            median_missing_rate=("missing_rate", "median"),
        )
        .sort_values("shap_share", ascending=False)
    )
    transformed = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "raw_feature": [transformed_to_raw(name)[0] for name in transformed_names],
            "is_missing_indicator": [transformed_to_raw(name)[1] for name in transformed_names],
            "gain": gains,
            "gain_share": gains / gains.sum(),
            "split_count": splits,
            "mean_abs_shap_log_odds": mean_abs_shap,
            "shap_share": mean_abs_shap / mean_abs_shap.sum(),
        }
    ).sort_values("mean_abs_shap_log_odds", ascending=False)
    pairs = pd.DataFrame(pair_rows).sort_values(
        "absolute_association", ascending=False, ignore_index=True
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "model_factor_audit.csv", index=False)
    transformed.to_csv(args.output_dir / "transformed_factor_audit.csv", index=False)
    pairs.to_csv(args.output_dir / "factor_correlation_pairs.csv", index=False)
    group_summary.to_csv(args.output_dir / "factor_group_summary.csv", index=False)

    top = audit.head(30).copy()
    top["SHAP share"] = top["shap_share"].map(lambda value: f"{value:.2%}")
    top["Gain share"] = top["gain_share"].map(lambda value: f"{value:.2%}")
    top["Missing"] = top["missing_rate"].map(lambda value: f"{value:.1%}")
    columns = {
        "feature": "Factor",
        "source_group": "Signal family",
        "SHAP share": "SHAP share",
        "Gain share": "Gain share",
        "observed_model_direction": "Observed direction",
        "Missing": "Missing",
        "strongest_related_feature": "Strongest related factor",
    }
    report = [
        "# CrediFast frozen-model factor audit",
        "",
        f"This audit covers all **{len(audit)} raw factors** and **{len(transformed)} transformed inputs** used by the frozen history LightGBM model on **{len(engineered):,} untouched holdout rows**. It is diagnostic only; the holdout was not used to refit, tune, calibrate, or select features.",
        "",
        "## How to read magnitude and direction",
        "",
        "- `total_mean_abs_shap_log_odds` is the average absolute raw-score contribution, including an automatically generated missingness indicator when present. It is the best global measure here of how strongly the fitted model moves because of a factor.",
        "- `gain_share` is LightGBM's training-time split-gain allocation. Correlated variables divide gain unpredictably, so use it alongside SHAP rather than alone.",
        "- `observed_model_direction` comes from the Spearman relationship between the factor and its own SHAP contribution. `upper_minus_lower_shap` and the event-rate difference show the marginal upper-versus-lower contrast.",
        "- SHAP values are in the history model's uncalibrated log-odds space. They explain model behavior, not causality or a legally sufficient lending reason.",
        "- `substitutability_assessment` is a redundancy screen. It is not permission to remove a variable; correlated factors can have distinct interactions, missingness, and stability.",
        "",
        "## Contribution by signal family",
        "",
        markdown_table(
            group_summary.assign(
                shap_share=group_summary["shap_share"].map(lambda value: f"{value:.2%}"),
                gain_share=group_summary["gain_share"].map(lambda value: f"{value:.2%}"),
                median_missing_rate=group_summary["median_missing_rate"].map(
                    lambda value: f"{value:.1%}"
                ),
            )
        ),
        "",
        "## Top 30 individual factors",
        "",
        markdown_table(top[list(columns)].rename(columns=columns)),
        "",
        "## Complete files",
        "",
        "- `model_factor_audit.csv`: all raw business factors, definitions, intuitive and observed directions, magnitudes, missingness, correlations, and substitute warnings.",
        "- `transformed_factor_audit.csv`: all direct model columns plus automatic missingness indicators.",
        "- `factor_correlation_pairs.csv`: all numeric Spearman or categorical Cramer's V associations at absolute association >= 0.50.",
        "- `factor_group_summary.csv`: contribution and missingness summarized by signal family.",
        "",
        "## Important limitations",
        "",
        "This is an observational audit of one fitted model on one random holdout from Home Credit. Marginal direction can differ from local direction because LightGBM is nonlinear and interactive. External-score definitions are anonymized. Categorical-to-numeric dependence is not summarized in the pair file. A factor-removal decision requires grouped ablation on a fresh development or out-of-time split.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "status": "diagnostic_only_frozen_holdout_not_retuned",
        "holdout_rows": len(engineered),
        "raw_factor_count": len(audit),
        "transformed_factor_count": len(transformed),
        "automatic_missing_indicator_count": int(transformed["is_missing_indicator"].sum()),
        "correlation_sample_rows": min(args.correlation_sample, len(engineered)),
        "tree_shap_space": "uncalibrated raw log odds",
        "files": [
            "model_factor_audit.csv",
            "transformed_factor_audit.csv",
            "factor_correlation_pairs.csv",
            "factor_group_summary.csv",
            "README.md",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
