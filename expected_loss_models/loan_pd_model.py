from typing import List

import mlflow.pyfunc
import numpy as np
import pandas as pd

PURPOSES = ["purchase", "refi", "cash_out", "other"]
REQUIRED_COLS = [
    "credit_score",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "loan_age_months",
    "original_principal_balance",
    "interest_rate",
    "employment_years",
    "delinquency_30d_past_12m",
    "loan_purpose",
    "tenor",
]

INPUT_ALIASES = {
    "fico": "credit_score",
    "dti": "debt_to_income_ratio",
    "ltv": "loan_to_value_ratio",
    "original_balance": "original_principal_balance",
    "employment_length_years": "employment_years",
    "delinquency_30d_12m": "delinquency_30d_past_12m",
    "tenor_years": "tenor",
    "pd_tenor": "tenor",
}

FEATURE_NAME_MAP = {
    "credit_score": "fico",
    "debt_to_income_ratio": "dti",
    "loan_to_value_ratio": "ltv",
    "loan_age_months": "loan_age_months",
    "original_principal_balance": "original_balance",
    "interest_rate": "interest_rate",
    "employment_years": "employment_length_years",
    "delinquency_30d_past_12m": "delinquency_30d_12m",
}


def _ensure_columns(model_input: pd.DataFrame, columns: List[str]) -> None:
    missing = set(columns) - set(model_input.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _encode_purpose(series: pd.Series) -> np.ndarray:
    categories = pd.Categorical(series, categories=PURPOSES)
    return categories.codes


def _apply_aliases(model_input: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for old, new in INPUT_ALIASES.items():
        if new not in model_input.columns and old in model_input.columns:
            rename_map[old] = new
    if not rename_map:
        return model_input
    return model_input.rename(columns=rename_map)


class LoanPDModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import xgboost as xgb

        self.model = xgb.XGBClassifier()
        self.model.load_model(context.artifacts["xgb_model"])

    def _extract_features(self, model_input: pd.DataFrame) -> pd.DataFrame:
        model_input = _apply_aliases(model_input)
        _ensure_columns(model_input, REQUIRED_COLS)
        features = model_input.copy()
        _coerce_numeric(
            features,
            [
                "credit_score",
                "debt_to_income_ratio",
                "loan_to_value_ratio",
                "loan_age_months",
                "original_principal_balance",
                "interest_rate",
                "employment_years",
                "delinquency_30d_past_12m",
            ],
        )
        features["loan_purpose_code"] = _encode_purpose(features["loan_purpose"])
        return pd.DataFrame(
            {
                FEATURE_NAME_MAP["credit_score"]: features["credit_score"],
                FEATURE_NAME_MAP["debt_to_income_ratio"]: features["debt_to_income_ratio"],
                FEATURE_NAME_MAP["loan_to_value_ratio"]: features["loan_to_value_ratio"],
                FEATURE_NAME_MAP["loan_age_months"]: features["loan_age_months"],
                FEATURE_NAME_MAP["original_principal_balance"]: features["original_principal_balance"],
                FEATURE_NAME_MAP["interest_rate"]: features["interest_rate"],
                FEATURE_NAME_MAP["employment_years"]: features["employment_years"],
                FEATURE_NAME_MAP["delinquency_30d_past_12m"]: features["delinquency_30d_past_12m"],
                "loan_purpose_code": features["loan_purpose_code"],
            }
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        model_input = _apply_aliases(model_input)
        _ensure_columns(model_input, REQUIRED_COLS)

        tenors = pd.to_numeric(model_input["tenor"], errors="coerce").values
        if (tenors[np.isfinite(tenors)] < 0).any():
            raise ValueError("tenor must be non-negative")

        features = self._extract_features(model_input)
        pds_1y = self.model.predict_proba(features)[:, 1]

        # Scale 1Y PD to requested tenor: PD(t) = 1 - (1 - PD_1y)^t
        pds = 1.0 - (1.0 - pds_1y) ** tenors

        return pd.Series(pds, name="probability_of_default")
