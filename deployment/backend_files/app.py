"""SuperKart API: validate engineered inputs and predict sales revenue."""

from pathlib import Path
import csv
import hashlib
import io
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

BASE_DIR = Path(__file__).resolve().parent

# Load documentation and input-support references shipped with the model.
with (BASE_DIR / "model_metadata.json").open(encoding="utf-8") as file:
    METADATA = json.load(file)

with (BASE_DIR / "feature_contract.json").open(encoding="utf-8") as file:
    CONTRACT = json.load(file)

with (BASE_DIR / "input_reference.json").open(encoding="utf-8") as file:
    REFERENCE = json.load(file)

MODEL_PATH = BASE_DIR / "superkart_model.joblib"

# Verify the artifact before loading the project's trusted serialized model.
hasher = hashlib.sha256()
with MODEL_PATH.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        hasher.update(chunk)

if hasher.hexdigest() != METADATA["artifact_sha256"]:
    raise RuntimeError("Model fingerprint does not match metadata.")

if REFERENCE["model_run_id"] != METADATA["run_id"]:
    raise RuntimeError("Input references belong to a different model.")

MODEL = joblib.load(MODEL_PATH)
FEATURES = CONTRACT["feature_columns"]
NUMERIC = CONTRACT["numeric_features"]
CATEGORICAL = CONTRACT["categorical_features"]

app = Flask(__name__)

# Bound upload size and batch length for the demonstration service.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
MAX_BATCH_ROWS = 5000


class InputValidationError(ValueError):
    """Represent a client input error rather than a model/server failure."""


def prepare_inputs(frame):
    """
    Validate engineered inputs and return ordered features plus warnings.

    Required values must be supplied. Unknown categorical strings are
    preserved for the pipeline's handle_unknown='ignore' encoder.
    """
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES))

    if missing or extra:
        raise InputValidationError(
            f"Column mismatch. Missing: {missing}; unexpected: {extra}."
        )

    if frame.empty or len(frame) > MAX_BATCH_ROWS:
        raise InputValidationError(
            f"Provide between 1 and {MAX_BATCH_ROWS} records."
        )

    data = frame[FEATURES].copy().reset_index(drop=True)

    # Reject missing values at the API boundary, even though the saved
    # pipeline contains imputers for its general preprocessing contract.
    if data.isna().any().any():
        raise InputValidationError("Required input values cannot be missing.")

    for column in NUMERIC:
        # JSON booleans are not valid measurements, despite Python treating
        # bool as an integer subtype.
        if data[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise InputValidationError(f"{column} must be numeric, not boolean.")

        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (ValueError, TypeError):
            raise InputValidationError(f"{column} must contain numbers.")

        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise InputValidationError(f"{column} must contain finite numbers.")

    for column in CATEGORICAL:
        if not data[column].map(lambda value: isinstance(value, str)).all():
            raise InputValidationError(f"{column} must contain text.")

        data[column] = data[column].str.strip()
        if data[column].eq("").any():
            raise InputValidationError(f"{column} cannot be blank.")

    # Apply the same documented sugar-label aliases as training.
    sugar_aliases = {
        "low sugar": "Low Sugar",
        "regular": "Regular",
        "reg": "Regular",
        "no sugar": "No Sugar",
    }
    sugar = data["Product_Sugar_Content"]
    data["Product_Sugar_Content"] = (
        sugar.str.casefold().map(sugar_aliases).fillna(sugar)
    )

    # Validate measurement meaning independently of training-set ranges.
    for column in ["Product_Weight", "Product_MRP"]:
        if data[column].le(0).any():
            raise InputValidationError(f"{column} must be greater than zero.")

    if not data["Product_Allocated_Area"].between(0, 1).all():
        raise InputValidationError("Product_Allocated_Area must be within [0, 1].")

    ages = data["Store_Age_Years"]
    if ages.lt(0).any() or ages.mod(1).ne(0).any():
        raise InputValidationError("Store_Age_Years must be a non-negative integer.")

    # Add row-level support warnings without changing the input profile.
    warnings = []
    profile_columns = REFERENCE["store_profile_columns"]
    known_profiles = {
        tuple(profile[column] for column in profile_columns)
        for profile in REFERENCE["observed_store_profiles"]
    }

    for row_number, row in data.iterrows():
        messages = []

        for column in CATEGORICAL:
            if row[column] not in REFERENCE["categorical_values"][column]:
                messages.append(f"Unseen category for {column}: {row[column]}.")

        for column in NUMERIC:
            bounds = REFERENCE["numeric_ranges"][column]
            if not bounds["minimum"] <= row[column] <= bounds["maximum"]:
                messages.append(f"{column} is outside the development-data range.")

        if tuple(row[column] for column in profile_columns) not in known_profiles:
            messages.append("Store profile was not represented in development data.")

        if messages:
            warnings.append({"row": int(row_number), "messages": messages})

    return data, warnings


def predict_frame(frame):
    """Run shared validation and vectorised prediction for either endpoint."""
    prepared, warnings = prepare_inputs(frame)
    predictions = np.asarray(MODEL.predict(prepared), dtype=float)

    if not np.isfinite(predictions).all():
        raise RuntimeError("Model returned non-finite predictions.")

    return {
        "model_version": METADATA["run_id"],
        "reference_year": CONTRACT["reference_year"],
        "predictions": predictions.tolist(),
        "warnings": warnings,
    }


@app.get("/health")
def health():
    """Report readiness after successful startup and model loading."""
    return jsonify(status="ok", model_version=METADATA["run_id"])


@app.post("/v1/predict")
def predict_single():
    """Accept one JSON object containing the ten engineered features."""
    payload = request.get_json()

    if not isinstance(payload, dict):
        raise InputValidationError("Provide one JSON object.")

    # Prevent nested lists/objects from becoming accidental DataFrame cells.
    if any(isinstance(value, (dict, list)) for value in payload.values()):
        raise InputValidationError("Each feature must contain one scalar value.")

    return jsonify(predict_frame(pd.DataFrame([payload])))


@app.post("/v1/predictbatch")
def predict_batch():
    """Accept a UTF-8 CSV upload under multipart field 'file'."""
    upload = request.files.get("file")

    if upload is None:
        raise InputValidationError("Upload a CSV using the field name 'file'.")

    try:
        text = upload.read().decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)), [])

        # pandas can rename duplicate headers automatically; reject them first.
        if len(header) != len(set(header)):
            raise InputValidationError("CSV column names must be unique.")

        frame = pd.read_csv(
            io.StringIO(text),
            nrows=MAX_BATCH_ROWS + 1,
        )
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error):
        raise InputValidationError("Provide a readable, non-empty UTF-8 CSV.")

    return jsonify(predict_frame(frame))


@app.errorhandler(InputValidationError)
def handle_input_error(error):
    """Return an actionable client error with a consistent JSON structure."""
    return jsonify(error=str(error)), 400


@app.errorhandler(HTTPException)
def handle_http_error(error):
    """Return JSON for malformed JSON, oversized uploads, and route errors."""
    return jsonify(error=error.description), error.code


@app.errorhandler(Exception)
def handle_server_error(error):
    """Log unexpected failures without exposing internal details to clients."""
    app.logger.exception("Unexpected prediction API failure")
    return jsonify(error="Prediction service encountered an internal error."), 500


if __name__ == "__main__":
    # Local development only; the container will use Gunicorn.
    app.run(host="0.0.0.0", port=7860, debug=False)
