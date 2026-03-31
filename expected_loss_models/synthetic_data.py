from typing import Tuple

import numpy as np
import pandas as pd

PURPOSES = ["purchase", "refi", "cash_out", "other"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_loans(n: int, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)

    fico = rng.normal(710, 60, n).clip(500, 850)
    dti = rng.beta(2, 5, n) * 0.5 + 0.1
    ltv = rng.beta(5, 2, n) * 0.5 + 0.5
    loan_age_months = rng.integers(1, 120, n)
    original_balance = rng.normal(250_000, 80_000, n).clip(50_000, 750_000)
    interest_rate = rng.normal(0.065, 0.01, n).clip(0.03, 0.12)
    employment_length_years = rng.integers(0, 30, n)
    delinquency_30d_12m = rng.poisson(0.2, n).clip(0, 5)
    loan_purpose = rng.choice(PURPOSES, n, p=[0.5, 0.3, 0.1, 0.1])
    loan_purpose_code = pd.Categorical(loan_purpose, categories=PURPOSES).codes

    score = (
        -5.0
        + (700 - fico) / 120
        + dti * 2.5
        + (ltv - 0.7) * 2.0
        + (interest_rate - 0.05) * 8.0
        + delinquency_30d_12m * 0.6
        + (loan_purpose_code == PURPOSES.index("cash_out")) * 0.5
    )
    noise = rng.normal(0, 0.5, n)
    prob = _sigmoid(score + noise)
    default = rng.binomial(1, prob, n)

    data = pd.DataFrame(
        {
            "fico": fico,
            "dti": dti,
            "ltv": ltv,
            "loan_age_months": loan_age_months,
            "original_balance": original_balance,
            "interest_rate": interest_rate,
            "employment_length_years": employment_length_years,
            "delinquency_30d_12m": delinquency_30d_12m,
            "loan_purpose_code": loan_purpose_code,
        }
    )
    return data, default
