import base64
import hashlib
import os
import re
import uuid

import numpy as np

import mlflow
import pandas as pd
import requests
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Array, ColSpec, Schema

from credit_curve_model import CreditCurveModel, build_credit_curve
from expected_loss_model import ExpectedLossModel
from loan_inventory_model import LoanInventoryModel, build_loan_inventory
from loan_pd_model import LoanPDModel
from train_pd_model import train_and_save_pd_model

BASE_DIR = os.path.dirname(__file__)
CODE_PATHS = [
    os.path.join(BASE_DIR, "credit_curve_model.py"),
    os.path.join(BASE_DIR, "expected_loss_model.py"),
    os.path.join(BASE_DIR, "loan_inventory_model.py"),
    os.path.join(BASE_DIR, "loan_pd_model.py"),
]


def _curve_examples():
    curve_input = pd.DataFrame({"curve_date": ["2024-12-31"]})
    curve_output = build_credit_curve("2024-12-31")
    return curve_input, curve_output


def _pd_examples(rng: np.random.Generator):
    row_count = int(rng.choice([2, 3, 4]))
    pd_input = pd.DataFrame(
        {
            "credit_score": rng.integers(620, 780, row_count).astype(float),
            "debt_to_income_ratio": rng.uniform(0.25, 0.55, row_count).astype(float),
            "loan_to_value_ratio": rng.uniform(0.7, 0.95, row_count).astype(float),
            "loan_age_months": rng.integers(6, 60, row_count).astype(float),
            "original_principal_balance": rng.integers(150000, 350000, row_count).astype(float),
            "interest_rate": rng.uniform(0.055, 0.085, row_count).astype(float),
            "employment_years": rng.integers(1, 10, row_count).astype(float),
            "delinquency_30d_past_12m": rng.integers(0, 2, row_count).astype(float),
            "loan_purpose": rng.choice(["purchase", "refi"], row_count),
            "tenor": rng.choice([1.0, 5.0, 10.0, 25.0], row_count),
        }
    )
    pd_output = pd.Series(
        rng.uniform(0.01, 0.1, row_count), name="probability_of_default"
    )
    return pd_input, pd_output


def _el_examples(
    pd_input: pd.DataFrame,
    curve_tenors: list[float],
    curve_rates: list[float],
    rng: np.random.Generator,
):
    row_count = len(pd_input)
    pd_1y_values = rng.uniform(0.001, 0.05, row_count)
    ltv_values = rng.uniform(0.60, 0.95, row_count)
    ead_values = rng.integers(150000, 350000, row_count).astype(float)
    el_input = pd.DataFrame(
        {
            "probability_of_default_1y": pd_1y_values,
            "probability_of_default_maturity": rng.uniform(0.05, 0.5, row_count),
            "loan_to_value_ratio": ltv_values,
            "current_balance": ead_values,
            "remaining_term_years": rng.uniform(1.0, 6.0, row_count).astype(float),
            "curve_tenors": [curve_tenors] * row_count,
            "curve_rates": [curve_rates] * row_count,
        }
    )
    lgd_approx = (ltv_values - 0.60).clip(0.05, 0.80)
    el_undisc = (
        el_input["probability_of_default_maturity"]
        * lgd_approx
        * el_input["current_balance"]
    ).round(2)
    el_disc = (el_undisc * 0.95).round(2)
    rwa = (ead_values * 0.35).round(2)
    el_output = pd.DataFrame(
        {
            "implied_credit_rating": ["BBB"] * row_count,
            "implied_lgd": list(lgd_approx.round(4)),
            "el_undiscounted": list(el_undisc),
            "el_discounted": list(el_disc),
            "rwa": list(rwa),
        }
    )
    return el_input, el_output


def _inventory_examples():
    inventory_input = pd.DataFrame(
        {"inventory_date": ["2024-12-31"], "number_of_items": [10]}
    )
    inventory_output = build_loan_inventory("2024-12-31", 10)
    return inventory_input, inventory_output


def _stringify_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = result[col].map(lambda val: "" if pd.isna(val) else str(val))
    return result


def domino_short_id(length: int = 8) -> str:
    def short_fallback() -> str:
        return base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("utf-8").rstrip("=")[:length]

    user = os.environ.get("DOMINO_PROJECT_OWNER") or short_fallback()
    project = os.environ.get("DOMINO_PROJECT_ID") or short_fallback()

    combined = f"{user}/{project}"
    digest = hashlib.sha256(combined.encode()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return f"{user}_{encoded[:length]}"


def _camel_case(name: str) -> str:
    if name and all(ch.isalnum() for ch in name):
        return name
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", name) if token]
    return "".join(token[:1].upper() + token[1:].lower() for token in tokens)


def _experiment_name(model_name: str) -> str:
    return f"{_camel_case(model_name)}_{domino_short_id()}"


def resolve_registered_model_version(model_name: str, model_info: object) -> int | None:
    version_value = getattr(model_info, "registered_model_version", None)
    if version_value is not None:
        try:
            return int(version_value)
        except (TypeError, ValueError):
            pass
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name)
        if versions:
            return max(int(v.version) for v in versions if v.version is not None)
    except mlflow.exceptions.MlflowException:
        return None
    return None


def resolve_domino_url() -> tuple[str, str]:
    url = os.environ.get("DOMINO_URL", "")
    if not url:
        url = os.environ.get("DOMINO_API_HOST", "")
    return url, "DOMINO_URL"


def resolve_environment_id(
    domino_url: str,
    api_key: str,
    environment_id: str,
    project_id: str,
) -> tuple[str | None, str]:
    if environment_id:
        return environment_id, "DOMINO_ENVIRONMENT_ID"

    if not (domino_url and api_key and project_id):
        return None, "missing DOMINO_URL/DOMINO_USER_API_KEY/DOMINO_PROJECT_ID"

    # Get the project's default environment ID from the Domino API
    headers = {"X-Domino-Api-Key": api_key}

    # Try /v4/projects/{id}/settings first
    for endpoint in [
        f"{domino_url}/v4/projects/{project_id}/settings",
        f"{domino_url}/api/projects/v1/projects/{project_id}/settings",
    ]:
        try:
            response = requests.get(endpoint, headers=headers, timeout=30)
            response.raise_for_status()
            settings = response.json()
            default_env = settings.get("defaultEnvironmentId")
            if default_env:
                return default_env, "project default environment"
        except requests.RequestException:
            continue

    return None, "no default environment found for project"


def normalize_endpoint_name(name: str) -> str:
    if not any(ch for ch in name if not ch.isalnum()):
        return name
    parts = []
    current = []
    for ch in name:
        if ch.isalnum():
            current.append(ch)
        elif current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def _find_model_api_id(
    domino_url: str,
    headers: dict,
    project_id: str,
    model_api_name: str,
) -> str | None:
    try:
        response = requests.get(
            f"{domino_url}/api/modelServing/v1/modelApis",
            params={"projectId": project_id, "name": model_api_name},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    payload = response.json()
    items = payload.get("items", [])
    for item in items:
        if item.get("name") == model_api_name and not item.get("archived", False):
            return item.get("id")
    return None


def _get_model_api_by_id(
    domino_url: str,
    headers: dict,
    model_api_id: str,
) -> dict | None:
    try:
        response = requests.get(
            f"{domino_url}/api/modelServing/v1/modelApis/{model_api_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None
    payload = response.json()
    return payload if isinstance(payload, dict) else None


def register_model_api_endpoint(
    model_api_name: str,
    registered_model_name: str,
    registered_model_version: int,
) -> None:
    domino_url, domino_url_source = resolve_domino_url()
    domino_url = domino_url.rstrip("/")
    api_key = os.environ.get("DOMINO_USER_API_KEY", "")
    project_id = os.environ.get("DOMINO_PROJECT_ID", "")
    environment_id = os.environ.get("DOMINO_ENVIRONMENT_ID", "")
    resolved_environment_id, env_source = resolve_environment_id(
        domino_url,
        api_key,
        environment_id,
        project_id,
    )

    if not (domino_url and api_key and project_id and resolved_environment_id):
        missing = [
            name
            for name, value in {
                "DOMINO_URL": domino_url,
                "DOMINO_USER_API_KEY": api_key,
                "DOMINO_PROJECT_ID": project_id,
                "DOMINO_ENVIRONMENT_ID": resolved_environment_id,
            }.items()
            if not value
        ]
        print(f"Skipping model API registration; missing: {', '.join(missing)}")
        return

    print(
        "Using Domino URL from "
        f"{domino_url_source} and environment from {env_source}."
    )
    prediction_dataset_id = os.environ.get("DOMINO_PREDICTION_DATASET_ID", "").strip()
    monitoring_enabled = bool(prediction_dataset_id)
    description = (
        f"Model API for registered model {registered_model_name} "
        f"v{registered_model_version}"
    )
    payload = {
        "name": model_api_name,
        "description": description,
        "environmentId": resolved_environment_id,
        "isAsync": False,
        "strictNodeAntiAffinity": False,
        "environmentVariables": [],
        "version": {
            "projectId": project_id,
            "environmentId": resolved_environment_id,
            "source": {
                "type": "Registry",
                "registeredModelName": registered_model_name,
                "registeredModelVersion": registered_model_version,
            },
            "logHttpRequestResponse": True,
            "monitoringEnabled": monitoring_enabled,
            "recordInvocation": True,
            "shouldDeploy": True,
            "description": description,
        },
    }
    if prediction_dataset_id:
        payload["version"]["predictionDatasetResourceId"] = prediction_dataset_id

    headers = {"X-Domino-Api-Key": api_key}
    existing_id = _find_model_api_id(
        domino_url,
        headers,
        project_id,
        model_api_name,
    )
    if existing_id:
        existing_api = _get_model_api_by_id(domino_url, headers, existing_id) or {}
        update_payload = {
            "name": model_api_name,
            "description": description,
            "environmentId": resolved_environment_id,
            "replicas": existing_api.get("replicas", 1),
        }
        if "hardwareTierId" in existing_api:
            update_payload["hardwareTierId"] = existing_api.get("hardwareTierId")
        if "resourceQuotaId" in existing_api:
            update_payload["resourceQuotaId"] = existing_api.get("resourceQuotaId")
        try:
            update_response = requests.put(
                f"{domino_url}/api/modelServing/v1/modelApis/{existing_id}",
                json=update_payload,
                headers=headers,
                timeout=30,
            )
            update_response.raise_for_status()
            print(f"Updated model API metadata for {model_api_name} (id={existing_id})")
        except requests.RequestException as exc:
            details = getattr(getattr(exc, "response", None), "text", "")
            print(
                f"Failed to update model API metadata for {model_api_name}: {exc}"
                f"{' - ' + details if details else ''}"
            )

        version_payload = payload["version"]
        version_payload["projectId"] = project_id
        version_payload["environmentId"] = resolved_environment_id
        try:
            response = requests.post(
                f"{domino_url}/api/modelServing/v1/modelApis/{existing_id}/versions",
                json=version_payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            version_info = response.json()
            version_id = version_info.get("id", "unknown")
            print(
                f"Registered model API version {version_id} for endpoint "
                f"{model_api_name} (id={existing_id})"
            )
        except requests.RequestException as exc:
            details = getattr(getattr(exc, "response", None), "text", "")
            print(
                f"Failed to create model API version for {model_api_name}: {exc}"
                f"{' - ' + details if details else ''}"
            )
        return

    try:
        response = requests.post(
            f"{domino_url}/api/modelServing/v1/modelApis",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        model_api = response.json()
        model_api_id = model_api.get("id", "unknown")
        print(f"Registered model API endpoint {model_api_name} (id={model_api_id})")
    except requests.RequestException as exc:
        details = getattr(getattr(exc, "response", None), "text", "")
        print(
            f"Failed to create model API endpoint {model_api_name}: {exc}"
            f"{' - ' + details if details else ''}"
        )


def register_models():
    rng = np.random.default_rng()
    model_path, pd_metrics, pd_params = train_and_save_pd_model(
        "pd_model.json",
        n_samples=int(rng.choice([25_000, 40_000, 50_000, 60_000, 75_000])),
        seed=None,
        return_stats=True,
    )

    curve_input, curve_output = _curve_examples()
    curve_signature = infer_signature(curve_input, curve_output)

    pd_input, pd_output = _pd_examples(np.random.default_rng(0))

    pd_numeric_cols = [
        "credit_score",
        "debt_to_income_ratio",
        "loan_to_value_ratio",
        "loan_age_months",
        "original_principal_balance",
        "interest_rate",
        "employment_years",
        "delinquency_30d_past_12m",
        "tenor",
    ]
    pd_input_example = _stringify_columns(pd_input, pd_numeric_cols)
    pd_signature = ModelSignature(
        inputs=Schema([ColSpec("string", name=col) for col in pd_input_example.columns]),
        outputs=Schema(
            [
                ColSpec("double", name="probability_of_default"),
            ]
        ),
    )

    curve_tenors = curve_output["years"].astype(float).tolist()
    curve_rates = curve_output["risk_free_rate"].astype(float).tolist()
    el_input, el_output = _el_examples(pd_input, curve_tenors, curve_rates, rng)
    el_numeric_cols = [
        "probability_of_default_1y",
        "probability_of_default_maturity",
        "loan_to_value_ratio",
        "current_balance",
        "remaining_term_years",
    ]
    el_input_example = _stringify_columns(el_input, el_numeric_cols)
    el_signature = ModelSignature(
        inputs=Schema(
            [
                ColSpec("string", name="probability_of_default_1y"),
                ColSpec("string", name="probability_of_default_maturity"),
                ColSpec("string", name="loan_to_value_ratio"),
                ColSpec("string", name="current_balance"),
                ColSpec("string", name="remaining_term_years"),
                ColSpec(Array("double"), name="curve_tenors"),
                ColSpec(Array("double"), name="curve_rates"),
            ]
        ),
        outputs=Schema(
            [
                ColSpec("string", name="implied_credit_rating"),
                ColSpec("double", name="implied_lgd"),
                ColSpec("double", name="el_undiscounted"),
                ColSpec("double", name="el_discounted"),
                ColSpec("double", name="rwa"),
            ]
        ),
    )

    inventory_input, inventory_output = _inventory_examples()
    inventory_signature = ModelSignature(
        inputs=Schema([
            ColSpec("string", name="inventory_date"),
            ColSpec("long", name="number_of_items"),
        ]),
        outputs=Schema(
            [
                ColSpec("string", name="loan_id"),
                ColSpec("double", name="credit_score"),
                ColSpec("double", name="debt_to_income_ratio"),
                ColSpec("double", name="loan_to_value_ratio"),
                ColSpec("double", name="loan_age_months"),
                ColSpec("double", name="original_principal_balance"),
                ColSpec("double", name="current_balance"),
                ColSpec("double", name="interest_rate"),
                ColSpec("double", name="employment_years"),
                ColSpec("double", name="delinquency_30d_past_12m"),
                ColSpec("string", name="loan_purpose"),
                ColSpec("double", name="original_loan_term_years"),
                ColSpec("double", name="remaining_term_years"),
            ]
        ),
    )

    curve_metrics = {
        "avg_risk_free_rate": float(curve_output["risk_free_rate"].mean()),
        "max_risk_free_rate": float(curve_output["risk_free_rate"].max()),
        "min_risk_free_rate": float(curve_output["risk_free_rate"].min()),
        "avg_discount_factor": float(curve_output["discount_factor"].mean()),
        "max_spread_bb": float(curve_output["spread_BB"].max()),
    }
    curve_params = {
        "curve_date": str(curve_input["curve_date"].iloc[0]),
        "tenor_count": float(len(curve_output)),
        "spread_ratings": float(len([c for c in curve_output.columns if c.startswith("spread_")])),
    }

    el_metrics = {
        "total_el_undiscounted": float(el_output["el_undiscounted"].sum()),
        "total_el_discounted": float(el_output["el_discounted"].sum()),
        "total_rwa": float(el_output["rwa"].sum()),
    }
    el_params = {
        "example_loan_count": float(len(el_input)),
        "curve_length": float(len(curve_tenors)),
    }

    inventory_params = {
        "inventory_date": str(inventory_input["inventory_date"].iloc[0]),
        "loan_count": float(len(inventory_output)),
    }

    mlflow.set_experiment(_experiment_name("GetCreditCurves"))
    with mlflow.start_run():
        mlflow.log_params(curve_params)
        mlflow.log_metrics(curve_metrics)
        curve_model_info = mlflow.pyfunc.log_model(
            name="GetCreditCurves",
            python_model=CreditCurveModel(),
            signature=curve_signature,
            input_example=curve_input,
            code_paths=CODE_PATHS,
            registered_model_name="GetCreditCurves",
        )

    mlflow.set_experiment(_experiment_name("GetLoanProbabilityOfDefault"))
    with mlflow.start_run():
        mlflow.log_params(pd_params)
        mlflow.log_metrics(pd_metrics)
        pd_model_info = mlflow.pyfunc.log_model(
            name="GetLoanProbabilityOfDefault",
            python_model=LoanPDModel(),
            artifacts={"xgb_model": model_path},
            signature=pd_signature,
            input_example=pd_input_example,
            code_paths=CODE_PATHS,
            registered_model_name="GetLoanProbabilityOfDefault",
        )

    mlflow.set_experiment(_experiment_name("GetExpectedLoss"))
    with mlflow.start_run():
        mlflow.log_params(el_params)
        mlflow.log_metrics(el_metrics)
        el_model_info = mlflow.pyfunc.log_model(
            name="GetExpectedLoss",
            python_model=ExpectedLossModel(),
            signature=el_signature,
            input_example=el_input_example,
            code_paths=CODE_PATHS,
            registered_model_name="GetExpectedLoss",
        )

    mlflow.set_experiment(_experiment_name("GetLoanInventory"))
    with mlflow.start_run():
        mlflow.log_params(inventory_params)
        inventory_model_info = mlflow.pyfunc.log_model(
            name="GetLoanInventory",
            python_model=LoanInventoryModel(),
            signature=inventory_signature,
            input_example=inventory_input,
            code_paths=CODE_PATHS,
            registered_model_name="GetLoanInventory",
        )

    for model_name, model_info in [
        ("GetCreditCurves", curve_model_info),
        ("GetLoanProbabilityOfDefault", pd_model_info),
        ("GetExpectedLoss", el_model_info),
        ("GetLoanInventory", inventory_model_info),
    ]:
        version = resolve_registered_model_version(model_name, model_info)
        if version is None:
            print(f"Skipping model API registration for {model_name}; no version found.")
            continue
        endpoint_name = normalize_endpoint_name(model_name)
        register_model_api_endpoint(
            model_api_name=endpoint_name,
            registered_model_name=model_name,
            registered_model_version=version,
        )


if __name__ == "__main__":
    register_models()
