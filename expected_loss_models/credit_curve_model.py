import hashlib
import io
from typing import Dict

import mlflow.pyfunc
import numpy as np
import pandas as pd

TENORS = [
    ("3M", 0.25),
    ("1Y", 1.0),
    ("2Y", 2.0),
    ("3Y", 3.0),
    ("5Y", 5.0),
    ("10Y", 10.0),
    ("30Y", 30.0),
]

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B"]


def _date_tweak(curve_date: str) -> float:
    digest = hashlib.md5(curve_date.encode("utf-8")).hexdigest()
    basis_points = (int(digest[:4], 16) % 41) - 20
    return basis_points / 10_000.0


def _risk_free_rate(years: float, tweak: float) -> float:
    base_years = np.array([0.25, 1, 2, 3, 5, 10, 30], dtype=float)
    base_rates = np.array([0.0365, 0.0347, 0.0347, 0.0355, 0.0374, 0.0419, 0.0445])
    rate = float(np.interp(years, base_years, base_rates))
    return max(rate + tweak, 0.001)


def _spread_tweak(curve_date: str, rating: str) -> float:
    digest = hashlib.md5(f"{curve_date}:{rating}".encode("utf-8")).hexdigest()
    basis_points = (int(digest[:4], 16) % 21) - 10
    return basis_points / 10_000.0


def _spreads(years: float, curve_date: str) -> Dict[str, float]:
    base = {
        "AAA": 0.0050,
        "AA": 0.0080,
        "A": 0.0120,
        "BBB": 0.0180,
        "BB": 0.0350,
        "B": 0.0550,
    }
    term_widen = {
        "AAA": 0.0003,
        "AA": 0.0004,
        "A": 0.0006,
        "BBB": 0.0009,
        "BB": 0.0012,
        "B": 0.0018,
    }
    return {
        rating: base[rating] + term_widen[rating] * years + _spread_tweak(curve_date, rating)
        for rating in RATINGS
    }


def build_credit_curve(curve_date: str) -> pd.DataFrame:
    tweak = _date_tweak(str(curve_date))
    rows = []
    for tenor, years in TENORS:
        rf_rate = _risk_free_rate(years, tweak)
        discount_factor = float(np.exp(-rf_rate * years))
        spreads = _spreads(years, curve_date)
        row = {
            "tenor": tenor,
            "years": years,
            "discount_factor": round(discount_factor, 6),
            "risk_free_rate": round(rf_rate, 6),
        }
        for rating in RATINGS:
            row[f"spread_{rating}"] = round(spreads[rating], 6)
        rows.append(row)
    return pd.DataFrame(rows)


def curve_to_json(curve_df: pd.DataFrame) -> str:
    return curve_df.to_json(orient="records")


def json_to_curve(curve_json: str) -> pd.DataFrame:
    return pd.read_json(io.StringIO(curve_json), orient="records")


class CreditCurveModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if "curve_date" not in model_input.columns:
            raise ValueError("Missing required column: curve_date")
        curve_date = str(model_input["curve_date"].iloc[0])
        return build_credit_curve(curve_date)
