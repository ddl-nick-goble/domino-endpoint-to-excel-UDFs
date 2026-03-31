from pathlib import Path

import pandas as pd

from credit_curve_model import CreditCurveModel
from expected_loss_model import ExpectedLossModel
from loan_pd_model import LoanPDModel
from train_pd_model import train_and_save_pd_model


class _DummyContext:
    def __init__(self, artifacts):
        self.artifacts = artifacts


def run_smoke_test():
    curve_in = pd.DataFrame({"curve_date": ["2024-12-31"]})
    curve_out = CreditCurveModel().predict(None, curve_in)
    curve_tenors = curve_out["years"].astype(float).tolist()
    curve_rates = curve_out["risk_free_rate"].astype(float).tolist()

    model_path = Path("pd_model.json")
    if not model_path.exists():
        train_and_save_pd_model(str(model_path))

    loan_features = pd.DataFrame(
        {
            "credit_score": [720, 650],
            "debt_to_income_ratio": [0.35, 0.48],
            "loan_to_value_ratio": [0.8, 0.92],
            "loan_age_months": [12, 36],
            "original_principal_balance": [250000, 180000],
            "interest_rate": [0.065, 0.075],
            "employment_years": [5, 2],
            "delinquency_30d_past_12m": [0, 1],
            "loan_purpose": ["purchase", "refi"],
        }
    )

    pd_model = LoanPDModel()
    pd_model.load_context(_DummyContext({"xgb_model": str(model_path)}))

    # Get 1Y PD
    loans_1y = loan_features.copy()
    loans_1y["tenor"] = 1.0
    pd_out_1y = pd_model.predict(None, loans_1y)

    # Get maturity PD
    remaining_terms = [4.5, 2.0]
    loans_mat = loan_features.copy()
    loans_mat["tenor"] = remaining_terms
    pd_out_mat = pd_model.predict(None, loans_mat)

    print("PD at 1Y:", pd_out_1y.tolist())
    print("PD at maturity:", pd_out_mat.tolist())

    el_in = pd.DataFrame(
        {
            "probability_of_default_1y": pd_out_1y.values,
            "probability_of_default_maturity": pd_out_mat.values,
            "loan_to_value_ratio": loan_features["loan_to_value_ratio"].values,
            "current_balance": [235000, 168000],
            "remaining_term_years": remaining_terms,
            "curve_tenors": [curve_tenors] * 2,
            "curve_rates": [curve_rates] * 2,
        }
    )

    el_out = ExpectedLossModel().predict(None, el_in)
    print(el_out)


if __name__ == "__main__":
    run_smoke_test()
