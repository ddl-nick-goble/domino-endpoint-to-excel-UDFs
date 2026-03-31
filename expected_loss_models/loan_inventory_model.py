from __future__ import annotations

import hashlib

import mlflow.pyfunc
import numpy as np
import pandas as pd


def _seed_from_date(inventory_date: str) -> int:
    return int(hashlib.md5(inventory_date.encode()).hexdigest()[:8], 16)


def _compute_current_balance(
    original_principal_balance: float,
    annual_interest_rate: float,
    original_loan_term_years: int,
    loan_age_months: int,
) -> float:
    """Compute remaining amortized balance using standard mortgage math."""
    n = original_loan_term_years * 12
    t = min(loan_age_months, n - 1)
    r = annual_interest_rate / 12.0
    if r == 0.0:
        remaining = max(1, n - t)
        return round(original_principal_balance * remaining / n, 2)
    factor_n = (1 + r) ** n
    factor_t = (1 + r) ** t
    return round(max(0.0, original_principal_balance * (factor_n - factor_t) / (factor_n - 1)), 2)


def _generate_loan(rng: np.random.Generator, loan_number: int) -> dict:
    loan_id = f"L{loan_number:03d}"

    # Credit score: normal ~720, std 40, clipped [620, 800], integer
    credit_score = int(np.clip(rng.normal(720, 40), 620, 800))

    # DTI: beta-shaped, range ~[0.15, 0.55], 2 decimal places
    debt_to_income_ratio = round(float(np.clip(rng.beta(2, 4) * 0.45 + 0.15, 0.15, 0.55)), 2)

    # LTV: beta-shaped skewed high, range ~[0.60, 0.97], 2 decimal places
    loan_to_value_ratio = round(float(np.clip(rng.beta(5, 2) * 0.37 + 0.60, 0.60, 0.97)), 2)

    # Loan age: uniform [6, 96] months, integer
    loan_age_months = int(rng.integers(6, 97))

    # Original balance: normal ~300k, std 80k, rounded to nearest 5000
    raw_balance = np.clip(rng.normal(300000, 80000), 100000, 600000)
    original_principal_balance = int(round(raw_balance / 5000) * 5000)

    # Interest rate: normal ~5.5%, std 1%, rounded to nearest 10bps
    interest_rate = round(float(np.clip(rng.normal(0.055, 0.01), 0.035, 0.085)), 3)

    # Employment years: exponential-ish, most people 1-8, tail to 25, integer
    employment_years = int(np.clip(rng.exponential(5) + 1, 1, 25))

    # Delinquency: mostly 0, Poisson lambda=0.15, clipped [0, 4], integer
    delinquency_30d_past_12m = int(np.clip(rng.poisson(0.15), 0, 4))

    # Loan purpose: categorical with weights
    loan_purpose = rng.choice(
        ["purchase", "refi"],
        p=[0.60, 0.40],
    )

    # Original term: categorical [15, 20, 25, 30] weighted toward 30
    original_loan_term_years = int(rng.choice(
        [15, 20, 25, 30],
        p=[0.10, 0.15, 0.20, 0.55],
    ))

    # Remaining term: original minus elapsed, at least 1
    elapsed_years = loan_age_months / 12.0
    remaining_term_years = max(1, int(original_loan_term_years - elapsed_years))

    current_balance = _compute_current_balance(
        original_principal_balance, interest_rate, original_loan_term_years, loan_age_months
    )

    return {
        "loan_id": loan_id,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "loan_to_value_ratio": loan_to_value_ratio,
        "loan_age_months": loan_age_months,
        "original_principal_balance": original_principal_balance,
        "interest_rate": interest_rate,
        "employment_years": employment_years,
        "delinquency_30d_past_12m": delinquency_30d_past_12m,
        "loan_purpose": loan_purpose,
        "original_loan_term_years": original_loan_term_years,
        "remaining_term_years": remaining_term_years,
        "current_balance": current_balance,
    }


ORDERED_COLS = [
    "loan_id",
    "loan_purpose",
    "credit_score",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "loan_age_months",
    "original_principal_balance",
    "current_balance",
    "interest_rate",
    "employment_years",
    "delinquency_30d_past_12m",
    "original_loan_term_years",
    "remaining_term_years",
]


def build_loan_inventory(
    inventory_date: str,
    number_of_items: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(_seed_from_date(inventory_date))
    rows = [_generate_loan(rng, i + 1) for i in range(number_of_items)]
    return pd.DataFrame(rows)[ORDERED_COLS]


class LoanInventoryModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if "inventory_date" not in model_input.columns:
            raise ValueError("Missing required column: inventory_date")
        inventory_date = str(model_input["inventory_date"].iloc[0])
        number_of_items = 10
        if "number_of_items" in model_input.columns:
            val = model_input["number_of_items"].iloc[0]
            if val is not None and not pd.isna(val):
                number_of_items = int(val)
        return build_loan_inventory(inventory_date, number_of_items)
