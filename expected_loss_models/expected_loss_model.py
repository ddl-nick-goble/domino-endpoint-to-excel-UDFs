from typing import List, Tuple

import json

import mlflow.pyfunc
import numpy as np
import pandas as pd

from credit_curve_model import RATINGS, _spreads

REQUIRED_COLS = [
    "probability_of_default_1y",
    "probability_of_default_maturity",
    "loan_to_value_ratio",
    "current_balance",
    "remaining_term_years",
    "curve_tenors",
    "curve_rates",
]

INPUT_ALIASES = {
    "pd_1y": "probability_of_default_1y",
    "pd_maturity": "probability_of_default_maturity",
    "ltv": "loan_to_value_ratio",
    "ead": "current_balance",
    "years_to_maturity": "remaining_term_years",
}

# Basel III standardized approach risk weights for residential mortgages (LTV-based)
LTV_RISK_WEIGHTS = [
    (0.50, 0.20),
    (0.60, 0.25),
    (0.80, 0.35),
    (0.90, 0.50),
    (1.00, 0.75),
    (float("inf"), 1.05),
]


def _ltv_risk_weight(ltv: float) -> float:
    for threshold, weight in LTV_RISK_WEIGHTS:
        if ltv <= threshold:
            return weight
    return 1.05

PD_RATING_THRESHOLDS = [
    (0.0004, "AAA"),
    (0.001, "AA"),
    (0.002, "A"),
    (0.005, "BBB"),
    (0.02, "BB"),
]


def _derive_credit_rating(pd_1y: float) -> str:
    """Map 1-year PD to an implied credit rating."""
    for threshold, rating in PD_RATING_THRESHOLDS:
        if pd_1y < threshold:
            return rating
    return "B"


def _derive_lgd(ltv: float) -> float:
    """Derive LGD from loan-to-value ratio.

    Lower LTV means more equity cushion and higher recovery on foreclosure.
    A loan at 60% LTV has near-zero loss; at 95% LTV roughly 35% loss.
    """
    return max(0.05, min(0.80, ltv - 0.60))


def _ensure_columns(model_input: pd.DataFrame, columns: List[str]) -> None:
    missing = set(columns) - set(model_input.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def _coerce_numeric(model_input: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        before_na = int(model_input[col].isna().sum()) if col in model_input.columns else -1
        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")
        after_na = int(model_input[col].isna().sum())
        print(f"[ExpectedLossModel] Coerce numeric '{col}': NaN before={before_na} after={after_na}")


def _apply_aliases(model_input: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for old, new in INPUT_ALIASES.items():
        if new not in model_input.columns and old in model_input.columns:
            rename_map[old] = new
    if not rename_map:
        return model_input
    print(f"[ExpectedLossModel] Applying aliases: {rename_map}")
    return model_input.rename(columns=rename_map)


def get_risky_discount_factor(
    curve_df: pd.DataFrame,
    rating: str,
    years: float,
) -> float:
    spread_col = f"spread_{rating}"
    if spread_col not in curve_df.columns:
        raise ValueError(f"Unsupported rating: {rating}")
    rf_rate = float(np.interp(years, curve_df["years"], curve_df["risk_free_rate"]))
    spread = float(np.interp(years, curve_df["years"], curve_df[spread_col]))
    risky_rate = rf_rate + spread
    return float(np.exp(-risky_rate * years))


def compute_expected_loss(
    row: pd.Series,
    curve_df: pd.DataFrame,
    implied_rating: str,
    implied_lgd: float,
) -> Tuple[float, float, float]:
    pd_maturity = row["probability_of_default_maturity"]
    ead = row["current_balance"]
    ltv = row["loan_to_value_ratio"]
    remaining_years = row["remaining_term_years"]

    el_undisc = pd_maturity * implied_lgd * ead
    df = get_risky_discount_factor(curve_df, implied_rating, remaining_years)
    el_disc = el_undisc * df
    rwa = ead * _ltv_risk_weight(ltv)
    if not np.isfinite([el_undisc, el_disc, rwa]).all():
        raise ValueError("Computed values contain NaN or Inf")
    return float(el_undisc), float(el_disc), float(rwa)


def _coerce_curve_array(value, label: str) -> np.ndarray:
    print(f"[ExpectedLossModel] Raw {label} type={type(value).__name__} value={value}")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {label} JSON array") from exc
    if value is None:
        raise ValueError(f"Missing {label}")
    arr = np.asarray(value, dtype=float)
    print(f"[ExpectedLossModel] Coerced {label} dtype={arr.dtype} shape={arr.shape} sample={arr[:5] if arr.size else arr}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Invalid {label}: contains NaN or Inf")
    return arr


def _curve_from_arrays(curve_tenors, curve_rates) -> pd.DataFrame:
    tenors = _coerce_curve_array(curve_tenors, "curve_tenors")
    rates = _coerce_curve_array(curve_rates, "curve_rates")
    if tenors.shape != rates.shape:
        raise ValueError("curve_tenors and curve_rates must have matching lengths")
    curve_date = "static"
    data = {"years": tenors, "risk_free_rate": rates}
    for rating in RATINGS:
        data[f"spread_{rating}"] = [
            _spreads(float(years), curve_date)[rating] for years in tenors
        ]
    return pd.DataFrame(data).sort_values("years")


class ExpectedLossModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        print(f"[ExpectedLossModel] Input columns: {list(model_input.columns)} rows={len(model_input)}")
        print(f"[ExpectedLossModel] Input head:\n{model_input.head(3)}")
        model_input = _apply_aliases(model_input)
        _ensure_columns(model_input, REQUIRED_COLS)
        _coerce_numeric(
            model_input,
            [
                "probability_of_default_1y",
                "probability_of_default_maturity",
                "loan_to_value_ratio",
                "current_balance",
                "remaining_term_years",
            ],
        )

        results = []
        for _, row in model_input.iterrows():
            curve_df = _curve_from_arrays(row["curve_tenors"], row["curve_rates"])
            implied_rating = _derive_credit_rating(row["probability_of_default_1y"])
            implied_lgd = _derive_lgd(row["loan_to_value_ratio"])
            print(f"[ExpectedLossModel] implied_rating={implied_rating} implied_lgd={implied_lgd:.4f} term={row['remaining_term_years']}")
            el_u, el_d, rwa = compute_expected_loss(row, curve_df, implied_rating, implied_lgd)
            results.append(
                {
                    "implied_credit_rating": implied_rating,
                    "implied_lgd": round(implied_lgd, 4),
                    "el_undiscounted": el_u,
                    "el_discounted": el_d,
                    "rwa": rwa,
                }
            )

        output = pd.DataFrame(results)
        print(f"[ExpectedLossModel] Output head:\n{output.head(3)}")
        return output
