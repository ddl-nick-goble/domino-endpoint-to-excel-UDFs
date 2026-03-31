#!/usr/bin/env python3
"""
Combined Excel-DNA Add-in Generator for Domino Model Endpoints

This script:
1. Fetches all models for a given Domino project
2. Extracts curl commands and credentials from the model overview pages
3. Retrieves parameter signatures from MLflow (if available)
4. Generates a complete Excel-DNA add-in (.xll) with User Defined Functions (UDFs)

Each discovered endpoint is converted into a strongly-typed UDF with full documentation,
parameter descriptions, and error handling.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import requests

# Environment configuration
DOMINO_URL = os.environ.get("DOMINO_URL", "")
API_KEY = os.environ.get("DOMINO_USER_API_KEY", "")
PROJECT_ID = os.environ.get("DOMINO_PROJECT_ID", "")
PROJECT_NAME = os.environ.get("DOMINO_PROJECT_NAME", "")


@dataclass
class EndpointConfig:
    """Configuration for a single API endpoint to be converted to a UDF."""
    name: str
    url: str
    parameters: list[dict[str, Any]]  # List of {name, type, description, example}
    description: str
    return_description: str


@dataclass
class GenAIEndpointConfig:
    """Configuration for a gen-AI (OpenAI-compatible) endpoint to be converted to a UDF."""
    name: str
    base_url: str  # e.g., "https://apps.se-demo.domino.tech/endpoints/{id}/v1"
    description: str


@dataclass
class AppAgentConfig:
    """Configuration for an agent hosted by a Domino AISYSTEM app.

    These agents are called via the app's REST API (POST /agents/{name})
    rather than directly through a GenAI /chat/completions endpoint.
    """
    agent_key: str           # Agent key used in the API route, e.g., "research_analyst"
    function_name: str       # C# method name, e.g., "ResearchAnalyst"
    display_name: str        # Human-readable name
    description: str         # Description for the Excel function tooltip
    app_url: str             # Base URL of the Domino app, e.g., "https://apps.host/apps/{vanityUrl}"
    app_name: str            # Name of the parent app


# =============================================================================
# Endpoint Discovery Functions (from claude_create_curls.py)
# =============================================================================

def get_models(project_id: str) -> list:
    """Get all models for a project."""
    url = f"{DOMINO_URL}/v4/modelManager/getModels"
    headers = {"X-Domino-Api-Key": API_KEY}
    resp = requests.get(url, params={"projectId": project_id}, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def discover_genai_endpoints(project_id: str, project_name: str) -> list[GenAIEndpointConfig]:
    """
    Discover gen-AI (OpenAI-compatible) endpoints from Domino App Endpoints.

    Queries /v4/modelProducts for running apps whose openUrl matches
    the /endpoints/{uuid}/ pattern, then builds GenAIEndpointConfig objects.
    """
    genai_endpoints = []

    # Build the apps domain: https://se-demo.domino.tech:443 -> https://apps.se-demo.domino.tech
    from urllib.parse import urlparse
    parsed = urlparse(DOMINO_URL)
    apps_base = f"{parsed.scheme}://apps.{parsed.hostname}"

    url = f"{DOMINO_URL}/v4/modelProducts"
    headers = {"X-Domino-Api-Key": API_KEY}
    try:
        resp = requests.get(url, params={"projectId": project_id}, headers=headers, timeout=30)
        resp.raise_for_status()
        products = resp.json()
    except Exception as e:
        print(f"  (could not query app endpoints: {e})")
        return []

    for product in products:
        open_url = product.get("openUrl", "")
        status = product.get("status", "")
        name = product.get("name", "UnnamedEndpoint")

        # Only running apps with /endpoints/{uuid}/ URL pattern
        if status != "Running":
            continue
        match = re.match(r'^/endpoints/([0-9a-f-]{36})/', open_url)
        if not match:
            continue

        endpoint_uuid = match.group(1)
        base_url = f"{apps_base}/endpoints/{endpoint_uuid}/v1"
        function_name = clean_function_name(name)

        genai_ep = GenAIEndpointConfig(
            name=function_name,
            base_url=base_url,
            description=f"Calls the {name} Domino GenAI endpoint.",
        )
        genai_endpoints.append(genai_ep)
        excel_name = f"Domino.LLM.{function_name}"
        print(f"    Found GenAI: {excel_name}(prompt, additional_context)")

    return genai_endpoints


def discover_agent_apps(project_id: str) -> list[AppAgentConfig]:
    """
    Discover agents hosted by AISYSTEM apps in a Domino project.

    1. Calls GET /api/apps/beta/apps?projectId={projectId} to find running apps
       with configurationType == "AISYSTEM".
    2. For each such app, calls GET {app_url}/agents to discover the agent catalog.
    3. Returns an AppAgentConfig for each discovered agent.
    """
    from urllib.parse import urlparse

    parsed = urlparse(DOMINO_URL)
    apps_base = f"{parsed.scheme}://apps.{parsed.hostname}"

    headers = {"X-Domino-Api-Key": API_KEY}

    # Step 1: Find AISYSTEM apps
    try:
        resp = requests.get(
            f"{DOMINO_URL}/api/apps/beta/apps",
            params={"projectId": project_id, "status": "Running"},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  (could not query apps API: {e})")
        return []

    items = data.get("items", [])
    aisystem_apps = [app for app in items if app.get("configurationType") == "AISYSTEM"]

    if not aisystem_apps:
        print("    No AISYSTEM apps found")
        return []

    # Step 2: For each AISYSTEM app, query its /agents endpoint
    agent_configs = []
    for app_info in aisystem_apps:
        vanity_url = app_info.get("vanityUrl", "")
        app_name = app_info.get("name", "UnknownApp")
        if not vanity_url:
            print(f"    Skipped app '{app_name}': no vanityUrl")
            continue

        app_url = f"{apps_base}/apps/{vanity_url}"
        print(f"    Found AISYSTEM app: {app_name} → {app_url}")

        # Query the agent catalog
        try:
            agents_resp = requests.get(
                f"{app_url}/agents",
                headers=headers,
                timeout=15,
            )
            agents_resp.raise_for_status()
            agents_catalog = agents_resp.json()
        except Exception as e:
            print(f"      Could not query agents from {app_name}: {e}")
            continue

        for agent_key, agent_info in agents_catalog.items():
            display_name = agent_info.get("display_name", agent_key)
            description = agent_info.get("description", f"Agent: {display_name}")
            function_name = clean_function_name(display_name)

            cfg = AppAgentConfig(
                agent_key=agent_key,
                function_name=function_name,
                display_name=display_name,
                description=description,
                app_url=app_url,
                app_name=app_name,
            )
            agent_configs.append(cfg)
            excel_name = f"Domino.Agent.{function_name}"
            print(f"      Agent: {excel_name}(prompt, additional_context)")

    return agent_configs


def _load_input_example(local_dir: str) -> dict | None:
    """Load the input example from MLflow artifacts, if present."""
    for fname in ("serving_input_example.json", "input_example.json"):
        path = os.path.join(local_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                example = json.load(f)
            # Convert dataframe_split format to simple dict
            if "dataframe_split" in example:
                cols = example["dataframe_split"]["columns"]
                data_rows = example["dataframe_split"]["data"]
                if not data_rows:
                    return {"data": {col: None for col in cols}}
                if len(data_rows) == 1:
                    row = data_rows[0]
                    return {"data": {col: row[i] for i, col in enumerate(cols)}}
                col_data = {col: [] for col in cols}
                for row in data_rows:
                    for i, col in enumerate(cols):
                        col_data[col].append(row[i])
                return {"data": col_data}
            if "data" in example:
                return example
    return None


def _load_signature_inputs(local_dir: str) -> list[dict[str, Any]] | None:
    """Load the MLflow model signature inputs from the MLmodel file."""
    mlmodel_path = os.path.join(local_dir, "MLmodel")
    if not os.path.exists(mlmodel_path):
        return None

    try:
        import yaml
    except Exception:
        return None

    with open(mlmodel_path) as f:
        mlmodel = yaml.safe_load(f)

    signature = mlmodel.get("signature") if isinstance(mlmodel, dict) else None
    if not signature:
        return None

    inputs = signature.get("inputs")
    if not inputs:
        return None

    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except json.JSONDecodeError:
            return None

    if isinstance(inputs, dict) and "inputs" in inputs:
        inputs = inputs["inputs"]

    if not isinstance(inputs, list):
        return None

    return inputs


def get_latest_registered_model_version(model_name: str) -> int | None:
    """Look up the latest version number for a registered MLflow model by name."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return None
    url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/registered-models/get-latest-versions"
    try:
        resp = requests.post(url, json={"name": model_name, "stages": []}, timeout=15)
        if resp.status_code != 200:
            return None
        versions = resp.json().get("model_versions", [])
        if not versions:
            return None
        return max(int(v["version"]) for v in versions if v.get("version"))
    except Exception:
        return None


def get_model_signature(model_name: str, model_version: int) -> dict | None:
    """Get the signature and input example for a registered model version from MLflow."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return None

    # Get the model version info to find the artifact source
    url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/model-versions/get"
    resp = requests.get(url, params={"name": model_name, "version": model_version}, timeout=15)
    if resp.status_code != 200:
        return None

    source = resp.json().get("model_version", {}).get("source")
    if not source:
        return None

    # Download the model artifacts and parse signature + input example
    try:
        from mlflow import artifacts
        os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
        local_dir = artifacts.download_artifacts(artifact_uri=source)

        signature_inputs = _load_signature_inputs(local_dir)
        example = _load_input_example(local_dir)
        if signature_inputs or example:
            return {"signature_inputs": signature_inputs, "example": example}
    except Exception:
        pass

    return None


def extract_help_topic_url(endpoint_url: str, domino_base_url: str) -> str:
    """Generate Domino model overview URL for HelpTopic."""
    match = re.search(r'/models/([a-f0-9]+)/', endpoint_url)
    if not match:
        return ""

    model_id = match.group(1)
    base_url = domino_base_url.rstrip(':443').rstrip('/')
    project_owner = os.environ.get("DOMINO_PROJECT_OWNER", "")
    project_name = os.environ.get("DOMINO_PROJECT_NAME", "")
    params = []
    if project_owner:
        params.append(f"ownerName={project_owner}")
    if project_name:
        params.append(f"projectName={project_name}")
    query = f"?{'&'.join(params)}" if params else ""
    return f"{base_url}/models/{model_id}/overview{query}"


def _split_param_tokens(name: str) -> list[str]:
    """Split a parameter name into lowercase tokens for heuristics."""
    if not name:
        return []
    tokens = re.findall(r"[A-Z]?[a-z]+|[0-9]+|[A-Z]+(?![a-z])", name)
    if not tokens:
        tokens = re.split(r"[^a-zA-Z0-9]+", name)
    return [t.lower() for t in tokens if t]


def _looks_like_date_string(value: str) -> bool:
    """Detect common date formats like yyyy-mm-dd or ISO timestamps."""
    if not value:
        return False
    return bool(re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}([ T].*)?$", value.strip()))


def _is_date_param(name: str, value: Any) -> bool:
    """Heuristic detection for date-like parameters."""
    tokens = _split_param_tokens(name)
    if any(t in {"date", "dt", "dob"} for t in tokens):
        return True
    if isinstance(value, str) and _looks_like_date_string(value):
        return True
    return False


def _map_mlflow_type(type_name: str, name: str, example: Any) -> str:
    """Map MLflow type name to a UDF parameter type."""
    if not type_name:
        return "string" if _is_date_param(name, example) else infer_parameter_type(example)

    lower = type_name.lower()
    if lower in {"boolean", "bool"}:
        return "bool"
    if lower in {"integer", "long", "int", "short"}:
        return "double"
    if lower in {"double", "float", "float32", "float64"}:
        return "double"
    if lower in {"date", "datetime"}:
        return "date"
    if lower in {"string", "str"}:
        return "date" if _is_date_param(name, example) else "string"
    return "object"


def _parse_mlflow_input_type(spec: dict[str, Any]) -> tuple[str, bool]:
    """Parse MLflow input spec to get base type and array flag."""
    type_name = str(spec.get("type", "") or "")

    if type_name.lower() == "tensor":
        tensor_spec = spec.get("tensor-spec", {}) or {}
        dtype = str(tensor_spec.get("dtype", "") or "")
        return _map_mlflow_type(dtype, spec.get("name", ""), None), True

    if type_name.lower() == "array":
        items = spec.get("items") or {}
        if isinstance(items, dict):
            item_type = str(items.get("type", "") or "")
        else:
            item_type = str(items)
        return _map_mlflow_type(item_type, spec.get("name", ""), None), True

    array_match = re.match(r"array\s*<\s*([^>]+)\s*>", type_name, flags=re.I)
    if not array_match:
        array_match = re.match(r"array\s*\(\s*([^)]+)\s*\)", type_name, flags=re.I)
    if not array_match:
        array_match = re.match(r"array\s*\[\s*([^\]]+)\s*\]", type_name, flags=re.I)
    if array_match:
        base_type = array_match.group(1)
        return _map_mlflow_type(base_type, spec.get("name", ""), None), True

    return _map_mlflow_type(type_name, spec.get("name", ""), None), False


def infer_parameter_type(value: Any) -> str:
    """Infer the C# type from a Python value."""
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "double"  # Use double for all numbers in Excel
    elif isinstance(value, float):
        return "double"
    elif isinstance(value, str):
        return "string"
    else:
        return "object"


def _should_allow_reference(param_type: str) -> bool:
    """
    Determine if a parameter should allow cell references.
    Numeric and date params can accept cell references for better UX.
    """
    return param_type in {"double", "date", "number"}


def _infer_array_element_type(value: Any) -> str:
    """Infer the element type for list/tuple values, handling nested lists."""
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple)):
                nested_type = _infer_array_element_type(item)
                if nested_type != "object":
                    return nested_type
            else:
                return infer_parameter_type(item)
    return "object"


def clean_function_name(name: str, prefix: str = "Model") -> str:
    """
    Clean a model name for use as a function name.
    Only applies CamelCase transformation if there's punctuation/spaces to remove.

    Examples:
        'hedging-model' -> 'HedgingModel'
        'my_cool_model' -> 'MyCoolModel'
        'HedgingModel' -> 'HedgingModel' (unchanged)
        'SimpleModel' -> 'SimpleModel' (unchanged)
    """
    # If the name is already clean (only alphanumeric), return as-is
    if re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', name):
        return name

    # Otherwise, split on non-alphanumeric and CamelCase it
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    camel = ''.join(part.capitalize() for part in parts if part)

    # Ensure it starts with a letter
    if camel and camel[0].isdigit():
        camel = prefix + camel
    return camel or f'Unnamed{prefix}'


def get_project_name(project_id: str) -> str:
    """Get the project name from environment variable or Domino API."""
    if PROJECT_NAME:
        return clean_function_name(PROJECT_NAME, prefix="Project")

    url = f"{DOMINO_URL}/v4/projects/{project_id}"
    headers = {"X-Domino-Api-Key": API_KEY}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        name = resp.json().get("name", "")
        if name:
            return clean_function_name(name, prefix="Project")
    except Exception:
        pass

    return ""


def discover_endpoints(project_id: str, project_name: str) -> tuple[list[EndpointConfig], list[GenAIEndpointConfig]]:
    """
    Discover all model endpoints in a project and build EndpointConfig objects.
    Returns (regular_endpoints, genai_endpoints).
    """
    endpoints = []
    models = get_models(project_id)

    for model in models:
        model_id = model.get("id")
        name = model.get("name", "UnnamedModel")
        active = model.get("activeVersion") or {}
        registered_name = active.get("registeredModelName")
        registered_version = active.get("registeredModelVersion")

        if not model_id:
            continue

        print(f"  Discovering: {name}...")

        # Build endpoint URL from API metadata
        endpoint_url = f"{DOMINO_URL}:443/models/{model_id}/latest/model"

        # Try to get the signature from MLflow.
        # If Domino's activeVersion doesn't carry registeredModelVersion (e.g. newly
        # deployed endpoints), fall back to looking up the latest version by name.
        signature = None
        if registered_name:
            version_to_use = registered_version or get_latest_registered_model_version(registered_name)
            if version_to_use:
                signature = get_model_signature(registered_name, version_to_use)

        if not signature:
            print(f"    (skipped - no MLflow signature found)")
            continue

        signature_inputs = signature.get("signature_inputs") or []
        example = signature.get("example")
        example_data = example.get("data") if isinstance(example, dict) else None

        parameters = []
        for spec in signature_inputs:
            param_name = spec.get("name")
            if not param_name:
                continue
            example_value = example_data.get(param_name) if isinstance(example_data, dict) else None
            param_type, is_array = _parse_mlflow_input_type(spec)
            if _is_date_param(param_name, example_value) and param_type == "string":
                param_type = "date"
            parameters.append({
                "name": param_name,
                "type": param_type,
                "is_array": is_array,
                "description": f"The {param_name} parameter for the model",
                "example": example_value
            })

        if not parameters:
            print(f"    (skipped - no parameters found in MLflow signature)")
            continue

        # Create the endpoint config
        # Use the endpoint name, cleaned up only if it has punctuation
        function_name = clean_function_name(name)
        endpoint = EndpointConfig(
            name=function_name,
            url=endpoint_url,
            parameters=parameters,
            description=f"Calls the {name} Domino model API endpoint.",
            return_description="Returns the model result value (spills across cells if array)"
        )
        endpoints.append(endpoint)
        param_names = ", ".join([p["name"] for p in parameters])
        excel_function_name = f"Domino.MLModel.{function_name}"
        print(f"    Found: {excel_function_name}({param_names})")

    return endpoints, []


# =============================================================================
# Code Generation Functions (from claude_create_udfs.py)
# =============================================================================

def generate_udf_method(endpoint: EndpointConfig, project_name: str) -> str:
    """Generate a C# UDF method for a single endpoint."""

    # Build ExcelArgument attributes and parameter section
    param_section_parts = []
    for p in endpoint.parameters:
        # Allow reference for array inputs or numeric/date params.
        allow_ref_str = "true" if p.get("is_array") or _should_allow_reference(p["type"]) else "false"
        excel_arg = f'[ExcelArgument(Name = "{p["name"]}", Description = "{p["description"]}", AllowReference = {allow_ref_str})]'
        param_type = "object"
        param_section_parts.append(f'{excel_arg} {param_type} {p["name"]}')

    param_section = ", ".join(param_section_parts)

    # Build JSON payload construction based on parameter types
    # We generate C# code that builds a proper JSON string with concatenation
    # Target output example for strings: "{\"data\": {\"key\": \"" + val.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"}}"
    # Target output example for numbers: "{\"data\": {\"key\": " + val.ToString(...) + "}}"
    json_parts = []
    for i, p in enumerate(endpoint.parameters):
        param_name = p["name"]
        is_last = (i == len(endpoint.parameters) - 1)
        separator = "" if is_last else ", "
        param_kind = p["type"]
        if param_kind == "object":
            param_kind = "string"
        if param_kind not in {"string", "bool", "date"}:
            param_kind = "number"
        kind_enum = {
            "string": "ParamKind.String",
            "bool": "ParamKind.Bool",
            "date": "ParamKind.Date",
            "number": "ParamKind.Number",
        }[param_kind]
        allow_array = "true" if p.get("is_array") else "false"
        json_parts.append(
            f'"\\\"{param_name}\\\": " + SerializeParamValue({param_name}, {kind_enum}, {allow_array}) + "{separator}"'
        )

    # Join the parts with + operator
    json_inner = " + ".join(json_parts)
    json_construction = f'"{{\\\"data\\\": {{" + {json_inner} + "}}}}"'

    # Escape description for C# string
    escaped_description = endpoint.description.replace('"', '\\"')

    # Excel function name with Domino. prefix
    excel_function_name = f"Domino.MLModel.{endpoint.name}"

    # Generate HelpTopic URL if possible
    help_topic_url = extract_help_topic_url(endpoint.url, DOMINO_URL)
    help_topic_line = f',\n            HelpTopic = "{help_topic_url}"' if help_topic_url else ''

    # Generate input validation: check each param for ExcelMissing/ExcelEmpty
    # Braces must be doubled because this is inserted into an f-string template
    validation_lines = []
    for p in endpoint.parameters:
        pn = p["name"]
        validation_lines.append(
            f'{{{{ var _v = NormalizeExcelValue({pn}); '
            f'if (_v == null || _v is ExcelMissing || _v is ExcelEmpty) '
            f'return "Error: Missing inputs."; }}}}'
        )
    input_validation = "\n                ".join(validation_lines)
    if input_validation:
        input_validation += "\n                "

    method = f'''
        /// <summary>
        /// {endpoint.description}
        /// </summary>
        /// <returns>{endpoint.return_description}</returns>
        [ExcelFunction(
            Name = "{excel_function_name}",
            Description = "{escaped_description}",
            Category = "Domino Model APIs",
            IsVolatile = false,
            IsExceptionSafe = true,
            IsThreadSafe = false{help_topic_line}
        )]
        public static object {endpoint.name}(
            {param_section})
        {{
            try
            {{
                {input_validation}string url = "{endpoint.url}";
                string jsonPayload = {json_construction};

                var pending = ExcelAsyncUtil.Observe("{excel_function_name}", new object[] {{ jsonPayload }}, () =>
                {{
                    return CallApiAsync(url, jsonPayload,
                        req => {{ }},
                        ParseResult
                    ).ToExcelObservable();
                }});
                if (pending is ExcelError && (ExcelError)pending == ExcelError.ExcelErrorNA)
                    return "Calling Domino...";
                return pending;
            }}
            catch (Exception ex)
            {{
                return "Error: " + ex.Message;
            }}
        }}
'''
    return method


def generate_genai_udf_method(endpoint: GenAIEndpointConfig, project_name: str) -> str:
    """Generate a C# UDF method for a gen-AI endpoint."""

    escaped_description = endpoint.description.replace('"', '\\"')

    excel_function_name = f"Domino.LLM.{endpoint.name}"

    # Build the JSON payload construction as C# code
    # Target C#: "{\"model\": \"\", \"messages\": [{\"role\": \"user\", \"content\": \"" + escapedPrompt + "\"}]}"
    json_construction = '"{\\\"model\\\": \\\"\\\", \\\"messages\\\": [{\\\"role\\\": \\\"user\\\", \\\"content\\\": \\\"" + escapedPrompt + "\\\"}]}"'

    method = f'''
        /// <summary>
        /// {endpoint.description}
        /// </summary>
        /// <returns>Returns the AI-generated response as a single string</returns>
        [ExcelFunction(
            Name = "{excel_function_name}",
            Description = "{escaped_description}",
            Category = "Domino AI LLMs",
            IsVolatile = false,
            IsExceptionSafe = true,
            IsThreadSafe = false
        )]
        public static object {endpoint.name}(
            [ExcelArgument(Name = "prompt", Description = "The prompt to send to the AI endpoint")] object prompt,
            [ExcelArgument(Name = "additional_context", Description = "Additional context data (array of any type, optional)")] object additional_context)
        {{
            try
            {{
                object normalizedPrompt = NormalizeExcelValue(prompt);
                if (normalizedPrompt == null || normalizedPrompt is ExcelMissing || normalizedPrompt is ExcelEmpty)
                    return "Enter a prompt to call the GenAI endpoint";
                string promptText = Convert.ToString(normalizedPrompt, CultureInfo.InvariantCulture);
                if (string.IsNullOrWhiteSpace(promptText))
                    return "Enter a prompt to call the GenAI endpoint";
                string contextText = SerializeAdditionalContext(additional_context);

                if (ContainsPendingPlaceholder(promptText) || ContainsPendingPlaceholder(contextText))
                    return "Waiting for dependencies...";

                string actualPrompt = promptText;
                if (!string.IsNullOrEmpty(contextText))
                {{
                    actualPrompt += "  The user added additional context: " + contextText;
                }}

                string url = "{endpoint.base_url}/chat/completions";
                string escapedPrompt = EscapeJsonString(actualPrompt);
                string jsonPayload = {json_construction};

                var pending = ExcelAsyncUtil.Observe("{excel_function_name}", new object[] {{ jsonPayload }}, () =>
                {{
                    return CallApiAsync(url, jsonPayload,
                        req => req.Headers.Add("X-Domino-Api-Key", "{API_KEY}"),
                        ParseGenAIResult
                    ).ToExcelObservable();
                }});
                if (pending is ExcelError && (ExcelError)pending == ExcelError.ExcelErrorNA)
                    return "Calling Domino...";
                return pending;
            }}
            catch (Exception ex)
            {{
                return "Error: " + ex.Message;
            }}
        }}
'''
    return method


def generate_app_agent_udf_method(agent: AppAgentConfig) -> str:
    """Generate a C# UDF method for an agent hosted by a Domino AISYSTEM app.

    Calls POST {app_url}/agents/{agent_key} with {"prompt": ..., "context": ...}
    and parses the {"response": ...} result.
    """

    escaped_description = agent.description.replace('"', '\\"')
    excel_function_name = f"Domino.Agent.{agent.function_name}"
    agent_url = f"{agent.app_url.rstrip('/')}/agents/{agent.agent_key}"

    method = f'''
        /// <summary>
        /// {agent.description}
        /// </summary>
        /// <returns>Returns the AI agent response as a single string</returns>
        [ExcelFunction(
            Name = "{excel_function_name}",
            Description = "{escaped_description}",
            Category = "Domino AI Agents",
            IsVolatile = false,
            IsExceptionSafe = true,
            IsThreadSafe = false
        )]
        public static object {agent.function_name}(
            [ExcelArgument(Name = "prompt", Description = "The prompt to send to the {agent.display_name} agent")] object prompt,
            [ExcelArgument(Name = "additional_context", Description = "Additional context data (array of any type, optional)")] object additional_context)
        {{
            try
            {{
                object normalizedPrompt = NormalizeExcelValue(prompt);
                if (normalizedPrompt == null || normalizedPrompt is ExcelMissing || normalizedPrompt is ExcelEmpty)
                    return "Enter a prompt to call {agent.display_name}";
                string promptText = Convert.ToString(normalizedPrompt, CultureInfo.InvariantCulture);
                if (string.IsNullOrWhiteSpace(promptText))
                    return "Enter a prompt to call {agent.display_name}";
                string contextText = SerializeAdditionalContext(additional_context);

                if (ContainsPendingPlaceholder(promptText) || ContainsPendingPlaceholder(contextText))
                    return "Waiting for dependencies...";

                string url = "{agent_url}";
                string escapedPrompt = EscapeJsonString(promptText);
                string jsonPayload;
                if (string.IsNullOrEmpty(contextText))
                {{
                    jsonPayload = "{{\\"prompt\\": \\"" + escapedPrompt + "\\"}}";
                }}
                else
                {{
                    string escapedContext = EscapeJsonString(contextText);
                    jsonPayload = "{{\\"prompt\\": \\"" + escapedPrompt + "\\", \\"context\\": \\"" + escapedContext + "\\"}}";
                }}

                var pending = ExcelAsyncUtil.Observe("{excel_function_name}", new object[] {{ jsonPayload }}, () =>
                {{
                    return CallApiAsync(url, jsonPayload,
                        req => req.Headers.Add("X-Domino-Api-Key", "{API_KEY}"),
                        ParseAgentResult
                    ).ToExcelObservable();
                }});
                if (pending is ExcelError && (ExcelError)pending == ExcelError.ExcelErrorNA)
                    return "Calling Domino...";
                return pending;
            }}
            catch (Exception ex)
            {{
                return "Error: " + ex.Message;
            }}
        }}
'''
    return method


def generate_csharp_code(endpoints: list[EndpointConfig], project_name: str,
                         genai_endpoints: list[GenAIEndpointConfig] | None = None,
                         app_agent_configs: list[AppAgentConfig] | None = None) -> str:
    """Generate the complete C# add-in code."""

    genai_endpoints = genai_endpoints or []
    app_agent_configs = app_agent_configs or []

    methods = "\n".join([generate_udf_method(ep, project_name) for ep in endpoints])
    genai_methods = "\n".join([generate_genai_udf_method(ep, project_name) for ep in genai_endpoints])
    app_agent_methods = "\n".join([generate_app_agent_udf_method(agent) for agent in app_agent_configs])

    # Build function documentation
    func_docs_parts = []
    for ep in endpoints:
        prefix = f"Domino.MLModel.{ep.name}"
        func_docs_parts.append(f"/// - {prefix}: {ep.description[:60]}...")
    for ep in genai_endpoints:
        prefix = f"Domino.LLM.{ep.name}"
        func_docs_parts.append(f"/// - {prefix}: {ep.description[:60]}...")
    for agent in app_agent_configs:
        prefix = f"Domino.Agent.{agent.function_name}"
        func_docs_parts.append(f"/// - {prefix}: {agent.description[:60]}...")
    func_docs = "\n".join(func_docs_parts)

    code = f'''using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using ExcelDna.Integration;

public static class TaskExcelObservableExtensions
{{
    public static IExcelObservable ToExcelObservable<T>(this Task<T> task)
    {{
        if (task == null) throw new ArgumentNullException(nameof(task));
        return new ExcelTaskObservable<T>(task);
    }}

    private class ExcelTaskObservable<T> : IExcelObservable
    {{
        private readonly Task<T> _task;
        public ExcelTaskObservable(Task<T> task) {{ _task = task; }}

        public IDisposable Subscribe(IExcelObserver observer)
        {{
            var cts = new CancellationTokenSource();
            switch (_task.Status)
            {{
                case TaskStatus.RanToCompletion:
                    if (!cts.IsCancellationRequested) {{ observer.OnNext(_task.Result); }}
                    break;
                case TaskStatus.Faulted:
                    if (!cts.IsCancellationRequested) observer.OnError(_task.Exception.InnerException);
                    break;
                case TaskStatus.Canceled:
                    if (!cts.IsCancellationRequested) observer.OnError(new TaskCanceledException(_task));
                    break;
                default:
                    _task.ContinueWith(t =>
                    {{
                        if (cts.IsCancellationRequested) return;
                        switch (t.Status)
                        {{
                            case TaskStatus.RanToCompletion:
                                observer.OnNext(t.Result);
                                break;
                            case TaskStatus.Faulted:
                                observer.OnError(t.Exception.InnerException);
                                break;
                            case TaskStatus.Canceled:
                                observer.OnError(new TaskCanceledException(t));
                                break;
                        }}
                    }}, TaskScheduler.Default);
                    break;
            }}
            return new CancellationDisposable(cts);
        }}
    }}

    private class CancellationDisposable : IDisposable
    {{
        private readonly CancellationTokenSource _cts;
        public CancellationDisposable(CancellationTokenSource cts) {{ _cts = cts; }}
        public void Dispose() {{ try {{ _cts.Cancel(); _cts.Dispose(); }} catch {{ }} }}
    }}
}}

/// <summary>
/// Initializes the Domino add-in: configures connection pooling and TLS settings once at load time.
/// </summary>
public class DominoAddIn : IExcelAddIn
{{
    public void AutoOpen()
    {{
        ExcelIntegration.RegisterUnhandledExceptionHandler(ex => "Error: " + ex.ToString());
        ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12 | SecurityProtocolType.Tls13;
        ServicePointManager.DefaultConnectionLimit = 4096;
        ServicePointManager.Expect100Continue = false;
        ExcelAsyncUtil.Initialize();
        System.Threading.ThreadPool.SetMinThreads(100, 100);
        try
        {{
            dynamic xlApp = ExcelDnaUtil.Application;
            xlApp.RTD.ThrottleInterval = 200;
        }}
        catch {{ }}
    }}

    public void AutoClose()
    {{
        ExcelAsyncUtil.Uninitialize();
    }}
}}

/// <summary>
/// Excel-DNA Add-in providing UDFs for Domino Model API endpoints.
///
/// This add-in was auto-generated and provides the following functions:
{func_docs}
///
/// Each function calls a specific Domino model endpoint with the provided parameters
/// and returns the result from the model (supports array spilling for multiple results).
/// </summary>
public static class DominoModelFunctions
{{
    private enum ParamKind
    {{
        String,
        Number,
        Bool,
        Date,
    }}

    private static string _cachedApiKey = null;
    private static readonly HttpClient _httpClient = new HttpClient();
    private static readonly SemaphoreSlim _requestThrottle = new SemaphoreSlim(4000, 4000);
    [ThreadStatic] private static Random _jitterRng;
    private static Random Jitter => _jitterRng ?? (_jitterRng = new Random(Guid.NewGuid().GetHashCode()));

    static DominoModelFunctions()
    {{
        _httpClient.Timeout = System.Threading.Timeout.InfiniteTimeSpan;
    }}

    /// <summary>
    /// Reads the Domino API key from environment variable or config file.
    /// Checks DOMINO_USER_API_KEY first, then DOMINO_API_KEY, then domino_config.json.
    /// </summary>
    private static string GetApiKey()
    {{
        if (_cachedApiKey != null) return _cachedApiKey;

        string envKey = Environment.GetEnvironmentVariable("DOMINO_USER_API_KEY");
        if (!string.IsNullOrEmpty(envKey))
        {{
            _cachedApiKey = envKey;
            return _cachedApiKey;
        }}

        envKey = Environment.GetEnvironmentVariable("DOMINO_API_KEY");
        if (!string.IsNullOrEmpty(envKey))
        {{
            _cachedApiKey = envKey;
            return _cachedApiKey;
        }}

        try
        {{
            string xllDir = Path.GetDirectoryName(ExcelDnaUtil.XllPath);
            string configPath = Path.Combine(xllDir, "domino_config.json");
            if (File.Exists(configPath))
            {{
                string json = File.ReadAllText(configPath);
                var match = Regex.Match(json, @"""api_key""\s*:\s*""([^""]+)""");
                if (match.Success)
                {{
                    _cachedApiKey = match.Groups[1].Value;
                    return _cachedApiKey;
                }}
            }}
        }}
        catch
        {{
            // Ignore file read errors
        }}

        return null;
    }}

    /// <summary>
    /// Serializes the additional_context parameter to a string for appending to prompts.
    /// Handles single values, 1D arrays, and 2D Excel ranges.
    /// </summary>
    private static string SerializeAdditionalContext(object value)
    {{
        value = NormalizeExcelValue(value);
        if (value == null || value is ExcelMissing || value is ExcelEmpty || value is ExcelError)
        {{
            return "";
        }}

        if (value is Array array)
        {{
            var sb = new StringBuilder();
            sb.Append('[');
            bool first = true;

            if (array.Rank == 1)
            {{
                for (int i = array.GetLowerBound(0); i <= array.GetUpperBound(0); i++)
                {{
                    object item = array.GetValue(i);
                    if (IsEmptyCell(item)) continue;
                    if (!first) sb.Append(", ");
                    sb.Append(Convert.ToString(item, CultureInfo.InvariantCulture));
                    first = false;
                }}
            }}
            else if (array.Rank == 2)
            {{
                for (int r = 0; r < array.GetLength(0); r++)
                {{
                    for (int c = 0; c < array.GetLength(1); c++)
                    {{
                        object item = array.GetValue(r, c);
                        if (IsEmptyCell(item)) continue;
                        if (!first) sb.Append(", ");
                        sb.Append(Convert.ToString(item, CultureInfo.InvariantCulture));
                        first = false;
                    }}
                }}
            }}

            sb.Append(']');
            return sb.ToString();
        }}

        return Convert.ToString(value, CultureInfo.InvariantCulture);
    }}

    /// <summary>
    /// Parses an OpenAI-compatible chat completion response to extract choices[0].message.content.
    /// </summary>
    private static string ParseGenAIResult(string json)
    {{
        int choicesIdx = json.IndexOf("\\\"choices\\\"");
        if (choicesIdx < 0)
            return "Error: No choices in GenAI response";

        int messageIdx = json.IndexOf("\\\"message\\\"", choicesIdx);
        if (messageIdx < 0)
            return "Error: No message in GenAI response";

        int contentIdx = json.IndexOf("\\\"content\\\"", messageIdx);
        if (contentIdx < 0)
            return "Error: No content in GenAI response";

        int colonIdx = json.IndexOf(':', contentIdx + 9);
        if (colonIdx < 0)
            return "Error: Malformed GenAI response";

        int i = colonIdx + 1;
        while (i < json.Length && char.IsWhiteSpace(json[i])) i++;

        if (i >= json.Length)
            return "Error: Empty content in GenAI response";

        if (json.Substring(i).StartsWith("null"))
            return "";

        if (json[i] != '"')
            return "Error: Unexpected content type in GenAI response";

        var sb = new StringBuilder();
        bool escape = false;
        for (int j = i + 1; j < json.Length; j++)
        {{
            char ch = json[j];
            if (escape)
            {{
                switch (ch)
                {{
                    case 'n': sb.Append('\\n'); break;
                    case 't': sb.Append('\\t'); break;
                    case 'r': sb.Append('\\r'); break;
                    case '"': sb.Append('"'); break;
                    case '\\\\': sb.Append('\\\\'); break;
                    case '/': sb.Append('/'); break;
                    default: sb.Append('\\\\'); sb.Append(ch); break;
                }}
                escape = false;
                continue;
            }}
            if (ch == '\\\\')
            {{
                escape = true;
                continue;
            }}
            if (ch == '"')
            {{
                return sb.ToString();
            }}
            sb.Append(ch);
        }}

        return "Error: Unterminated content string in GenAI response";
    }}

    /// <summary>
    /// Parses an agent app response to extract the "response" field.
    /// Expected format: {{"response": "...", "agent": "...", ...}}
    /// </summary>
    private static string ParseAgentResult(string json)
    {{
        int responseIdx = json.IndexOf("\\\"response\\\"");
        if (responseIdx < 0)
        {{
            // Check for error field
            int errorIdx = json.IndexOf("\\\"error\\\"");
            if (errorIdx >= 0)
            {{
                int eColon = json.IndexOf(':', errorIdx + 7);
                if (eColon >= 0)
                {{
                    int ei = eColon + 1;
                    while (ei < json.Length && char.IsWhiteSpace(json[ei])) ei++;
                    if (ei < json.Length && json[ei] == '"')
                    {{
                        var esb = new StringBuilder();
                        for (int ej = ei + 1; ej < json.Length; ej++)
                        {{
                            if (json[ej] == '\\\\') {{ ej++; if (ej < json.Length) esb.Append(json[ej]); continue; }}
                            if (json[ej] == '"') break;
                            esb.Append(json[ej]);
                        }}
                        return "Agent error: " + esb.ToString();
                    }}
                }}
            }}
            return "Error: No response in agent result";
        }}

        int colonIdx = json.IndexOf(':', responseIdx + 10);
        if (colonIdx < 0)
            return "Error: Malformed agent response";

        int i = colonIdx + 1;
        while (i < json.Length && char.IsWhiteSpace(json[i])) i++;

        if (i >= json.Length)
            return "Error: Empty response from agent";

        if (json.Substring(i).StartsWith("null"))
            return "";

        if (json[i] != '"')
            return "Error: Unexpected response type from agent";

        var sb = new StringBuilder();
        bool escape = false;
        for (int j = i + 1; j < json.Length; j++)
        {{
            char ch = json[j];
            if (escape)
            {{
                switch (ch)
                {{
                    case 'n': sb.Append('\\n'); break;
                    case 't': sb.Append('\\t'); break;
                    case 'r': sb.Append('\\r'); break;
                    case '"': sb.Append('"'); break;
                    case '\\\\': sb.Append('\\\\'); break;
                    case '/': sb.Append('/'); break;
                    default: sb.Append('\\\\'); sb.Append(ch); break;
                }}
                escape = false;
                continue;
            }}
            if (ch == '\\\\')
            {{
                escape = true;
                continue;
            }}
            if (ch == '"')
            {{
                return sb.ToString();
            }}
            sb.Append(ch);
        }}

        return "Error: Unterminated response string from agent";
    }}

    /// <summary>
    /// Formats a date-like parameter into yyyy-MM-dd.
    /// Accepts Excel dates, Unix epoch (seconds/ms), and date strings.
    /// </summary>
    private static string FormatDateParam(object value)
    {{
        if (value == null)
        {{
            return "";
        }}
        if (value is ExcelMissing || value is ExcelEmpty)
        {{
            return "";
        }}

        if (value is double d)
        {{
            return FormatDateFromNumber(d);
        }}
        if (value is int i)
        {{
            return FormatDateFromNumber(i);
        }}
        if (value is DateTime dt)
        {{
            return dt.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        }}
        if (value is string s)
        {{
            s = s.Trim();
            if (string.IsNullOrEmpty(s))
            {{
                return "";
            }}

            if (double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out double num))
            {{
                return FormatDateFromNumber(num);
            }}

            if (DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out DateTime parsed))
            {{
                return parsed.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
            }}

            return s;
        }}

        return value.ToString();
    }}

    private static string FormatDateFromNumber(double value)
    {{
        // Epoch milliseconds or seconds
        if (value >= 1_000_000_000_000d)
        {{
            var dt = DateTimeOffset.FromUnixTimeMilliseconds((long)Math.Round(value)).DateTime;
            return dt.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        }}
        if (value >= 1_000_000_000d)
        {{
            var dt = DateTimeOffset.FromUnixTimeSeconds((long)Math.Round(value)).DateTime;
            return dt.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        }}

        try
        {{
            var dt = DateTime.FromOADate(value);
            return dt.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        }}
        catch
        {{
            return value.ToString(CultureInfo.InvariantCulture);
        }}
    }}

    private static string EscapeJsonString(string value)
    {{
        if (value == null)
        {{
            return "";
        }}
        var sb = new System.Text.StringBuilder(value.Length);
        foreach (char c in value)
        {{
            switch (c)
            {{
                case '\\\\': sb.Append("\\\\\\\\"); break;
                case '\\"': sb.Append("\\\\\\""); break;
                case '\\n': sb.Append("\\\\n"); break;
                case '\\r': sb.Append("\\\\r"); break;
                case '\\t': sb.Append("\\\\t"); break;
                case '\\b': sb.Append("\\\\b"); break;
                case '\\f': sb.Append("\\\\f"); break;
                default:
                    if (c < ' ')
                        sb.AppendFormat("\\\\u{{0:X4}}", (int)c);
                    else
                        sb.Append(c);
                    break;
            }}
        }}
        return sb.ToString();
    }}

    private static object NormalizeExcelValue(object value)
    {{
        if (value is ExcelReference excelRef)
        {{
            try
            {{
                value = excelRef.GetValue();
            }}
            catch
            {{
                return null;
            }}
        }}
        return value;
    }}

    /// <summary>
    /// Returns true when text contains a pending-agent placeholder.
    /// Used to break recalculation loops when agent UDFs are chained:
    /// if an input still shows a placeholder we wait instead of firing
    /// a new API call that would cascade into another recalc cycle.
    /// </summary>
    private static bool ContainsPendingPlaceholder(string text)
    {{
        if (string.IsNullOrEmpty(text)) return false;
        return text.IndexOf("Calling Domino...", StringComparison.Ordinal) >= 0
            || text.IndexOf("Waiting for dependencies...", StringComparison.Ordinal) >= 0;
    }}

    private static string SerializeParamValue(object value, ParamKind kind, bool allowArray)
    {{
        value = NormalizeExcelValue(value);
        if (value == null || value is ExcelMissing || value is ExcelEmpty)
        {{
            return "null";
        }}

        if (allowArray && value is Array array)
        {{
            return SerializeArrayValue(array, kind);
        }}

        if (!allowArray && value is Array singleCell && singleCell.Rank == 2 && singleCell.GetLength(0) == 1 && singleCell.GetLength(1) == 1)
        {{
            return SerializeScalarValue(singleCell.GetValue(0, 0), kind);
        }}

        return SerializeScalarValue(value, kind);
    }}

    private static bool IsEmptyCell(object value)
    {{
        return value == null || value is ExcelMissing || value is ExcelEmpty || value is ExcelError;
    }}

    private static string SerializeArrayValue(Array array, ParamKind kind)
    {{
        if (array.Rank == 1)
        {{
            var sb = new StringBuilder();
            sb.Append('[');
            int start = array.GetLowerBound(0);
            int end = array.GetUpperBound(0);
            bool appended = false;
            for (int i = start; i <= end; i++)
            {{
                object value = array.GetValue(i);
                if (IsEmptyCell(value))
                {{
                    continue;
                }}
                if (appended)
                {{
                    sb.Append(',');
                }}
                sb.Append(kind == ParamKind.Number ? SerializeScalarValue(Convert.ToString(value, CultureInfo.InvariantCulture), ParamKind.String) : SerializeScalarValue(value, kind));
                appended = true;
            }}
            sb.Append(']');
            return sb.ToString();
        }}

        if (array.Rank == 2)
        {{
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            var sb = new StringBuilder();
            sb.Append('[');

            if (rows == 1 || cols == 1)
            {{
                int count = rows == 1 ? cols : rows;
                bool appended = false;
                for (int i = 0; i < count; i++)
                {{
                    object value = rows == 1 ? array.GetValue(0, i) : array.GetValue(i, 0);
                    if (IsEmptyCell(value))
                    {{
                        continue;
                    }}
                    if (appended)
                    {{
                        sb.Append(',');
                    }}
                    sb.Append(kind == ParamKind.Number ? SerializeScalarValue(Convert.ToString(value, CultureInfo.InvariantCulture), ParamKind.String) : SerializeScalarValue(value, kind));
                    appended = true;
                }}
                sb.Append(']');
                return sb.ToString();
            }}

            for (int r = 0; r < rows; r++)
            {{
                if (r > 0)
                {{
                    sb.Append(',');
                }}
                sb.Append('[');
                bool appended = false;
                for (int c = 0; c < cols; c++)
                {{
                    object value = array.GetValue(r, c);
                    if (IsEmptyCell(value))
                    {{
                        continue;
                    }}
                    if (appended)
                    {{
                        sb.Append(',');
                    }}
                    sb.Append(kind == ParamKind.Number ? SerializeScalarValue(Convert.ToString(value, CultureInfo.InvariantCulture), ParamKind.String) : SerializeScalarValue(value, kind));
                    appended = true;
                }}
                sb.Append(']');
            }}
            sb.Append(']');
            return sb.ToString();
        }}

        return SerializeScalarValue(array, kind);
    }}

    private static string SerializeScalarValue(object value, ParamKind kind)
    {{
        if (value == null || value is ExcelMissing || value is ExcelEmpty)
        {{
            return "null";
        }}

        switch (kind)
        {{
            case ParamKind.String:
                return "\\"" + EscapeJsonString(Convert.ToString(value, CultureInfo.InvariantCulture)) + "\\"";
            case ParamKind.Date:
                return "\\"" + EscapeJsonString(FormatDateParam(value)) + "\\"";
            case ParamKind.Bool:
                if (value is bool b)
                {{
                    return b ? "true" : "false";
                }}
                if (value is double d)
                {{
                    return d != 0d ? "true" : "false";
                }}
                if (value is int i)
                {{
                    return i != 0 ? "true" : "false";
                }}
                if (value is string s)
                {{
                    if (bool.TryParse(s, out bool parsedBool))
                    {{
                        return parsedBool ? "true" : "false";
                    }}
                    if (double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out double parsedNum))
                    {{
                        return parsedNum != 0d ? "true" : "false";
                    }}
                }}
                return "false";
            default:
                return SerializeNumberValue(value, false);
        }}
    }}

    private static string SerializeNumberValue(object value, bool forceFloat)
    {{
        if (value is double num)
        {{
            return FormatNumber(num, forceFloat);
        }}
        if (value is int numInt)
        {{
            return FormatNumber(numInt, forceFloat);
        }}
        if (value is string str)
        {{
            if (double.TryParse(str, NumberStyles.Any, CultureInfo.InvariantCulture, out double parsed))
            {{
                return FormatNumber(parsed, forceFloat);
            }}
            return "\\"" + EscapeJsonString(str) + "\\"";
        }}
        try
        {{
            return FormatNumber(Convert.ToDouble(value, CultureInfo.InvariantCulture), forceFloat);
        }}
        catch
        {{
            return "\\"" + EscapeJsonString(Convert.ToString(value, CultureInfo.InvariantCulture)) + "\\"";
        }}
    }}

    private static string FormatNumber(double value, bool forceFloat)
    {{
        if (!forceFloat)
        {{
            return value.ToString(CultureInfo.InvariantCulture);
        }}
        if (Math.Abs(value % 1) < 1e-12)
        {{
            return value.ToString("0.0", CultureInfo.InvariantCulture);
        }}
        return value.ToString(CultureInfo.InvariantCulture);
    }}

    /// <summary>
    /// Extracts the "result" field from the JSON response and returns it as Excel-friendly output.
    /// Handles single values, 1D arrays (horizontal spill), and 2D arrays (grid spill).
    /// </summary>
    private static object ParseResult(string json)
    {{
        if (!TryExtractResultValue(json, out string resultValue, out string error))
        {{
            return error;
        }}

        // Check if it's a 2D array (array of arrays) like [[1,2],[3,4]]
        if (resultValue.StartsWith("[["))
        {{
            return Parse2DArray(resultValue);
        }}
        // Check if it's a 1D array like [1,2,3]
        else if (resultValue.StartsWith("[") && resultValue.EndsWith("]"))
        {{
            return Parse1DArray(resultValue);
        }}
        else
        {{
            // Single value
            return ParseSingleValue(resultValue);
        }}
    }}

    /// <summary>
    /// Extracts the raw JSON value for the "result" field without relying on regex for nested arrays.
    /// </summary>
    private static bool TryExtractResultValue(string json, out string resultValue, out string error)
    {{
        resultValue = "";
        error = "";

        var match = Regex.Match(json, @"""result""\s*:");
        if (!match.Success)
        {{
            error = "Error: No result field in response";
            return false;
        }}

        int i = match.Index + match.Length;
        while (i < json.Length && char.IsWhiteSpace(json[i]))
        {{
            i++;
        }}

        if (i >= json.Length)
        {{
            error = "Error: Empty result field in response";
            return false;
        }}

        char start = json[i];
        if (start == '[' || start == '{{')
        {{
            char open = start;
            char close = (start == '[') ? ']' : '}}';
            int depth = 0;
            bool inString = false;
            bool escape = false;
            int startIndex = i;

            for (; i < json.Length; i++)
            {{
                char ch = json[i];
                if (inString)
                {{
                    if (escape)
                    {{
                        escape = false;
                        continue;
                    }}
                    if (ch == '\\\\')
                    {{
                        escape = true;
                        continue;
                    }}
                    if (ch == '"')
                    {{
                        inString = false;
                    }}
                    continue;
                }}

                if (ch == '"')
                {{
                    inString = true;
                    continue;
                }}

                if (ch == open)
                {{
                    depth++;
                }}
                else if (ch == close)
                {{
                    depth--;
                    if (depth == 0)
                    {{
                        resultValue = json.Substring(startIndex, i - startIndex + 1).Trim();
                        return true;
                    }}
                }}
            }}

            error = "Error: Unterminated result value in response";
            return false;
        }}

        if (start == '"')
        {{
            int startIndex = i;
            bool escape = false;
            for (i = i + 1; i < json.Length; i++)
            {{
                char ch = json[i];
                if (escape)
                {{
                    escape = false;
                    continue;
                }}
                if (ch == '\\\\')
                {{
                    escape = true;
                    continue;
                }}
                if (ch == '"')
                {{
                    resultValue = json.Substring(startIndex, i - startIndex + 1).Trim();
                    return true;
                }}
            }}
            error = "Error: Unterminated string result in response";
            return false;
        }}

        int primitiveStart = i;
        while (i < json.Length && json[i] != ',' && json[i] != '}}' && json[i] != ']')
        {{
            i++;
        }}
        resultValue = json.Substring(primitiveStart, i - primitiveStart).Trim();
        return true;
    }}

    /// <summary>
    /// Parses a 2D array like [[1,2,3],[4,5,6]] into an Excel-compatible object[,] for grid spill.
    /// </summary>
    private static object Parse2DArray(string arrayStr)
    {{
        // Extract inner arrays using regex to find each [...] row
        var rowMatches = Regex.Matches(arrayStr, @"\[([^\[\]]*)\]");
        if (rowMatches.Count == 0)
        {{
            return "Error: Invalid 2D array format";
        }}

        // Parse each row to get dimensions and values
        var rows = new List<List<object>>();
        int maxCols = 0;

        foreach (Match rowMatch in rowMatches)
        {{
            string rowContent = rowMatch.Groups[1].Value;
            var rowValues = new List<object>();

            if (!string.IsNullOrWhiteSpace(rowContent))
            {{
                var parts = rowContent.Split(new[] {{ ',' }}, StringSplitOptions.RemoveEmptyEntries);
                foreach (var part in parts)
                {{
                    rowValues.Add(ParseSingleValue(part.Trim()));
                }}
            }}

            rows.Add(rowValues);
            if (rowValues.Count > maxCols)
            {{
                maxCols = rowValues.Count;
            }}
        }}

        // Handle edge cases
        if (rows.Count == 0 || maxCols == 0)
        {{
            return "";
        }}
        if (rows.Count == 1 && rows[0].Count == 1)
        {{
            return rows[0][0];
        }}

        // Create 2D array for Excel grid spill
        object[,] spillArray = new object[rows.Count, maxCols];
        for (int r = 0; r < rows.Count; r++)
        {{
            for (int c = 0; c < maxCols; c++)
            {{
                if (c < rows[r].Count)
                {{
                    spillArray[r, c] = rows[r][c];
                }}
                else
                {{
                    spillArray[r, c] = ""; // Pad jagged arrays
                }}
            }}
        }}
        return spillArray;
    }}

    /// <summary>
    /// Parses a 1D array like [1,2,3] into an Excel-compatible object[,] for horizontal spill.
    /// </summary>
    private static object Parse1DArray(string arrayStr)
    {{
        string inner = arrayStr.Substring(1, arrayStr.Length - 2).Trim();
        if (string.IsNullOrEmpty(inner))
        {{
            return "";
        }}

        var parts = inner.Split(new[] {{ ',' }}, StringSplitOptions.RemoveEmptyEntries);
        var results = new List<object>();

        foreach (var part in parts)
        {{
            results.Add(ParseSingleValue(part.Trim()));
        }}

        if (results.Count == 1)
        {{
            return results[0];
        }}

        // Create a 1-row, N-column array for horizontal spill
        object[,] spillArray = new object[1, results.Count];
        for (int i = 0; i < results.Count; i++)
        {{
            spillArray[0, i] = results[i];
        }}
        return spillArray;
    }}

    /// <summary>
    /// Parses a single value (number or string) into the appropriate type.
    /// </summary>
    private static object ParseSingleValue(string value)
    {{
        string trimmed = value.Trim();
        if (double.TryParse(trimmed, System.Globalization.NumberStyles.Any,
            System.Globalization.CultureInfo.InvariantCulture, out double numVal))
        {{
            return numVal;
        }}
        return trimmed.Trim('"');
    }}

    private static async Task<object> CallApiAsync(
        string url, string jsonPayload,
        Action<HttpRequestMessage> configureAuth,
        Func<string, object> parseResponse)
    {{
        await Task.Delay(Jitter.Next(0, 200)).ConfigureAwait(false);
        await _requestThrottle.WaitAsync().ConfigureAwait(false);
        try
        {{
            int maxAttempts = 12;
            string lastError = null;
            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {{
                try
                {{
                    using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(120)))
                    {{
                        var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
                        var request = new HttpRequestMessage(HttpMethod.Post, url);
                        request.Content = content;
                        configureAuth(request);
                        using (var resp = await _httpClient.SendAsync(request, cts.Token).ConfigureAwait(false))
                        {{
                            string responseText = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                            if (resp.IsSuccessStatusCode)
                                return parseResponse(responseText);
                            int code = (int)resp.StatusCode;
                            lastError = "HTTP " + code;
                            if (code < 500)
                                return "API Error (HTTP " + code + "): " + responseText;
                        }}
                    }}
                }}
                catch (OperationCanceledException)
                {{
                    lastError = "timeout";
                }}
                catch (Exception ex)
                {{
                    lastError = ex.Message;
                }}
                if (attempt < maxAttempts - 1)
                    await Task.Delay(Jitter.Next(200, 1000 * (attempt + 1))).ConfigureAwait(false);
            }}
            return "Error after " + maxAttempts + " attempts (last: " + lastError + ")";
        }}
        finally
        {{
            _requestThrottle.Release();
        }}
    }}

{methods}
{genai_methods}
{app_agent_methods}
}}
'''
    return code


def generate_dna_file(project_name: str) -> str:
    """Generate the Excel-DNA .dna configuration file."""
    addin_name = "Domino endpoint UDFs for project"
    if project_name:
        addin_name = f"{addin_name} - {project_name}"
    return '''<?xml version="1.0" encoding="utf-8"?>
<DnaLibrary Name="''' + addin_name + '''" RuntimeVersion="v4.0">
  <ExternalLibrary Path="DominoModelFunctions.dll" ExplicitExports="false" LoadFromBytes="true" Pack="true" />
</DnaLibrary>
'''


def build_addin(endpoints: list[EndpointConfig], project_name: str,
                genai_endpoints: list[GenAIEndpointConfig] | None = None,
                app_agent_configs: list[AppAgentConfig] | None = None) -> str | None:
    """Build the Excel-DNA add-in."""

    genai_endpoints = genai_endpoints or []
    app_agent_configs = app_agent_configs or []

    if not endpoints and not genai_endpoints and not app_agent_configs:
        print("No endpoints to build. Exiting.")
        return None

    print()
    print("=" * 60)
    print("Building Excel Add-in")
    print("=" * 60)
    print()

    # Create a temporary build directory
    build_dir = tempfile.mkdtemp(prefix="exceldna_build_")
    print(f"[1/6] Created temporary build directory: {build_dir}")

    try:
        # Write the C# code
        cs_file = os.path.join(build_dir, "DominoModelFunctions.cs")
        total_udfs = len(endpoints) + len(genai_endpoints) + len(app_agent_configs)
        with open(cs_file, "w") as f:
            f.write(generate_csharp_code(endpoints, project_name, genai_endpoints, app_agent_configs))
        print(f"[2/6] Generated C# source code with {total_udfs} UDF(s):")
        for ep in endpoints:
            params = ", ".join([p["name"] for p in ep.parameters])
            function_name = f"Domino.MLModel.{ep.name}"
            print(f"       - {function_name}({params})")
        for ep in genai_endpoints:
            function_name = f"Domino.LLM.{ep.name}"
            print(f"       - {function_name}(prompt, additional_context) [GenAI]")
        for agent in app_agent_configs:
            function_name = f"Domino.Agent.{agent.function_name}"
            print(f"       - {function_name}(prompt, additional_context) [Agent]")

        # Write the .dna file
        dna_file = os.path.join(build_dir, "DominoModelFunctions.dna")
        with open(dna_file, "w") as f:
            f.write(generate_dna_file(project_name))
        print("[3/6] Generated Excel-DNA configuration file")

        # Create a .csproj file for building
        csproj_content = '''<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net48</TargetFramework>
    <OutputType>Library</OutputType>
    <GenerateAssemblyInfo>false</GenerateAssemblyInfo>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="ExcelDna.AddIn" Version="1.7.0" />
  </ItemGroup>
  <ItemGroup>
    <!-- These are part of .NET Framework 4.8 but not auto-referenced in SDK-style projects -->
    <Reference Include="System.Net.Http" />
    <Reference Include="Microsoft.CSharp" />
    <Reference Include="System.Drawing" />
  </ItemGroup>
</Project>
'''
        csproj_file = os.path.join(build_dir, "DominoModelFunctions.csproj")
        with open(csproj_file, "w") as f:
            f.write(csproj_content)
        print("[4/6] Generated project file")

        # Run dotnet restore and build
        print("[5/6] Building add-in (this may take a moment)...")

        # Restore packages
        result = subprocess.run(
            ["dotnet", "restore"],
            cwd=build_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"       Restore output: {result.stdout}")
            print(f"       Restore errors: {result.stderr}")
            raise RuntimeError(f"dotnet restore failed: {result.stderr}")

        # Build the project
        result = subprocess.run(
            ["dotnet", "build", "-c", "Release"],
            cwd=build_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"       Build output: {result.stdout}")
            print(f"       Build errors: {result.stderr}")
            raise RuntimeError(f"dotnet build failed: {result.stderr}")

        print("       Build completed successfully!")

        # Find the PACKED .xll files
        publish_dir = os.path.join(build_dir, "bin", "Release", "net48", "publish")

        src_xll_64 = None
        src_xll_32 = None

        if os.path.exists(publish_dir):
            for f in os.listdir(publish_dir):
                if f.endswith("-packed.xll"):
                    full_path = os.path.join(publish_dir, f)
                    if "64" in f:
                        src_xll_64 = full_path
                    else:
                        src_xll_32 = full_path

        if not src_xll_64 and not src_xll_32:
            # Fallback: search entire build directory for packed xll files
            for root, dirs, files in os.walk(build_dir):
                for f in files:
                    if f.endswith("-packed.xll"):
                        full_path = os.path.join(root, f)
                        if "64" in f:
                            src_xll_64 = full_path
                        else:
                            src_xll_32 = full_path

        if not src_xll_64 and not src_xll_32:
            raise RuntimeError("Could not find packed .xll files. Check build output.")

        # Copy to current directory
        copied_files = []
        if src_xll_64:
            dest_xll_64 = os.path.join(os.getcwd(), "DominoModelFunctions-AddIn64.xll")
            shutil.copy(src_xll_64, dest_xll_64)
            copied_files.append(("64-bit", dest_xll_64))

        if src_xll_32:
            dest_xll_32 = os.path.join(os.getcwd(), "DominoModelFunctions-AddIn.xll")
            shutil.copy(src_xll_32, dest_xll_32)
            copied_files.append(("32-bit", dest_xll_32))

        print(f"[6/6] Add-in created successfully!")
        for arch, path in copied_files:
            print(f"       {arch}: {path}")

        # Write source files for reference
        cs_output = os.path.join(os.getcwd(), "DominoModelFunctions.cs")
        dna_output = os.path.join(os.getcwd(), "DominoModelFunctions.dna")
        with open(cs_output, "w") as f:
            f.write(generate_csharp_code(endpoints, project_name, genai_endpoints, app_agent_configs))
        with open(dna_output, "w") as f:
            f.write(generate_dna_file(project_name))

        print()
        print("=" * 60)
        print("SUCCESS! Your Excel add-in is ready.")
        print("=" * 60)
        print()
        print("To use the add-in:")
        print("  1. Open Excel")
        print("  2. Go to File > Options > Add-ins")
        print("  3. At the bottom, select 'Excel Add-ins' and click 'Go...'")
        print("  4. Click 'Browse...' and select the .xll file")
        print("  5. Click OK to enable the add-in")
        print()
        print("Available functions:")
        for ep in endpoints:
            params = ", ".join([p["name"] for p in ep.parameters])
            function_name = f"Domino.MLModel.{ep.name}"
            print(f"  ={function_name}({params})")
            print(f"    {ep.description[:70]}...")
            print()
        for ep in genai_endpoints:
            function_name = f"Domino.LLM.{ep.name}"
            print(f"  ={function_name}(prompt, additional_context)")
            print(f"    {ep.description[:70]}...")
            print()
        for agent in app_agent_configs:
            function_name = f"Domino.Agent.{agent.function_name}"
            print(f"  ={function_name}(prompt, additional_context)")
            print(f"    {agent.description[:70]}...")
            print()

        artifacts_dir = "/mnt/artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        source_64 = "/mnt/code/DominoModelFunctions-AddIn64.xll"
        source_32 = "/mnt/code/DominoModelFunctions-AddIn.xll"
        dest_64 = os.path.join(artifacts_dir, "DominoExcelUDFsAddIn64.xll")
        dest_32 = os.path.join(artifacts_dir, "DominoExcelUDFsAddIn.xll")
        if os.path.exists(source_64):
            shutil.copy(source_64, dest_64)
        else:
            print(f"Warning: missing source file {source_64}")
        if os.path.exists(source_32):
            shutil.copy(source_32, dest_32)
        else:
            print(f"Warning: missing source file {source_32}")

        return copied_files[0][1] if copied_files else None

    finally:
        # Clean up the temp directory
        shutil.rmtree(build_dir, ignore_errors=True)



def main():
    """Main entry point - discover endpoints and build add-in."""

    include_raw_genai = True

    print("=" * 60)
    print("Domino Model APIs - Combined Endpoint Discovery & Add-in Generator")
    print("=" * 60)
    print()

    # Check required environment variables
    if not API_KEY:
        print("Error: DOMINO_USER_API_KEY environment variable is required")
        return

    project_id = PROJECT_ID
    if not project_id:
        print("Error: DOMINO_PROJECT_ID environment variable is required")
        return

    print(f"Domino URL: {DOMINO_URL}")
    print(f"Project ID: {project_id}")
    print()

    project_name = get_project_name(project_id)

    # Step 1: Discover endpoints
    print("Step 1: Discovering model endpoints...")
    print("-" * 40)
    endpoints, genai_from_models = discover_endpoints(project_id, project_name)

    print()
    print("Step 1b: Discovering GenAI app endpoints...")
    print("-" * 40)
    genai_from_apps = discover_genai_endpoints(project_id, project_name)

    # Merge gen-AI endpoints from both sources (model discovery + app discovery)
    genai_endpoints = genai_from_models + genai_from_apps

    # Step 1c: Discover agents from AISYSTEM apps
    print()
    print("Step 1c: Discovering agents from AISYSTEM apps...")
    print("-" * 40)
    app_agent_configs = discover_agent_apps(project_id)

    # Filter out raw GenAI UDFs if disabled
    if not include_raw_genai and genai_endpoints:
        print()
        print("  Raw GenAI UDFs disabled — skipping direct endpoint functions.")
        genai_endpoints = []

    if not endpoints and not genai_endpoints and not app_agent_configs:
        print()
        print("No valid endpoints discovered. Check that:")
        print("  - The project has deployed models with active versions")
        print("  - The models have input signatures (from MLflow or curl data)")
        print("  - Your API key has access to view the models")
        print("  - The project has running AISYSTEM apps with agents")
        return

    total = len(endpoints) + len(genai_endpoints) + len(app_agent_configs)
    print()
    print(f"Discovered {total} endpoint(s) ({len(endpoints)} MLModel, {len(genai_endpoints)} GenAI, {len(app_agent_configs)} Agent)")
    print()

    # Step 2: Build the add-in
    print("Step 2: Building Excel add-in...")
    print("-" * 40)

    try:
        build_addin(endpoints, project_name, genai_endpoints, app_agent_configs)
    except Exception as e:
        print(f"\nBuild Error: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure .NET SDK (6.0+) is installed: dotnet --version")
        print("  - On Linux/Mac, you may need to build on Windows for .xll generation")
        print()

        # Write fallback artifacts to disk for manual builds
        cs_output = os.path.join(os.getcwd(), "DominoModelFunctions.cs")
        dna_output = os.path.join(os.getcwd(), "DominoModelFunctions.dna")
        with open(cs_output, "w") as f:
            f.write(generate_csharp_code(endpoints, project_name, genai_endpoints, app_agent_configs))
        with open(dna_output, "w") as f:
            f.write(generate_dna_file(project_name))
        print(f"Fallback files written:")
        print(f"  - {cs_output}")
        print(f"  - {dna_output}")


if __name__ == "__main__":
    main()
