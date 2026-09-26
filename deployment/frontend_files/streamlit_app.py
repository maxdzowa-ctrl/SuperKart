"""SuperKart interface for single and batch revenue predictions."""

from pathlib import Path
import io
import json
import os

import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# In Docker Compose, "backend" resolves to the Flask service.
BACKEND_URL = os.environ.get(
    "BACKEND_URL", "http://backend:7860"
).rstrip("/")

with (BASE_DIR / "feature_contract.json").open(encoding="utf-8") as file:
    CONTRACT = json.load(file)

with (BASE_DIR / "input_reference.json").open(encoding="utf-8") as file:
    REFERENCE = json.load(file)

st.set_page_config(page_title="SuperKart Sales Prediction", layout="wide")
st.title("SuperKart Sales Prediction")
st.caption(
    "Estimate product–store revenue using the evaluated Random Forest model."
)
st.info(
    "Revenue uses the dataset's currency and reporting period, which were "
    "not specified. These predictions are not validated next-quarter forecasts."
)


def call_api(endpoint, **kwargs):
    """Call Flask and return JSON, displaying clear failures in the UI."""
    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            timeout=120,
            **kwargs,
        )
    except requests.RequestException:
        st.error("The prediction service is unavailable. Please try again.")
        return None

    try:
        result = response.json()
    except ValueError:
        st.error("The prediction service returned an unreadable response.")
        return None

    if not response.ok:
        st.error(result.get("error", "The prediction request failed."))
        return None

    return result


def show_warnings(result):
    """Display backend warnings using human-readable, one-based row numbers."""
    warnings = result.get("warnings", [])

    if not warnings:
        st.success("No input-support warnings were reported.")
        return

    st.warning(
        f"{len(warnings)} record(s) contain inputs outside observed "
        "development support. Prediction accuracy for these inputs is uncertain."
    )
    records = [
        {
            "Data row": warning["row"] + 1,
            "Warning": " ".join(warning["messages"]),
        }
        for warning in warnings
    ]
    st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)


single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction"])

with single_tab:
    st.subheader("Product and store details")

    profiles = REFERENCE["observed_store_profiles"]

    with st.form("single_prediction"):
        left, right = st.columns(2)

        with left:
            product_type = st.selectbox(
                "Product type",
                sorted(CONTRACT["product_category_map"]),
            )
            weight = st.number_input(
                "Product weight (dataset units)",
                min_value=0.01,
                value=12.65,
                step=0.1,
            )
            mrp = st.number_input(
                "Maximum retail price",
                min_value=0.01,
                value=147.15,
                step=1.0,
            )
            sugar = st.selectbox(
                "Sugar content",
                ["Low Sugar", "Regular", "No Sugar"],
            )

        with right:
            allocated_area = st.number_input(
                "Allocated display-area ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.056,
                step=0.001,
                format="%.3f",
                help="For example, 0.056 represents 5.6% of store display area.",
            )
            profile_index = st.selectbox(
                "Observed store profile",
                options=list(range(len(profiles))),
                format_func=lambda index: (
                    f"{profiles[index]['Store_Type']} | "
                    f"{profiles[index]['Store_Location_City_Type']} | "
                    f"{profiles[index]['Store_Size']}"
                ),
            )
            profile = profiles[profile_index]
            st.caption(
                f"Store age as of {CONTRACT['reference_year']}: "
                f"{profile['Store_Age_Years']} years."
            )

        submitted = st.form_submit_button("Predict revenue")

    if submitted:
        # Prefix mapping follows the development-data audit.
        if product_type in ["Hard Drinks", "Soft Drinks"]:
            prefix = "DR"
        elif product_type in ["Health and Hygiene", "Household", "Others"]:
            prefix = "NC"
        else:
            prefix = "FD"

        payload = {
            "Product_Weight": weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Size": profile["Store_Size"],
            "Store_Location_City_Type": profile["Store_Location_City_Type"],
            "Store_Type": profile["Store_Type"],
            "Product_Id_char": prefix,
            "Store_Age_Years": profile["Store_Age_Years"],
            "Product_Type_Category": CONTRACT["product_category_map"][product_type],
        }

        with st.spinner("Calculating prediction..."):
            result = call_api("/v1/predict", json=payload)

        if result is not None:
            st.metric("Predicted revenue", f"{result['predictions'][0]:,.2f}")
            show_warnings(result)
            st.caption(f"Model version: {result['model_version']}")

with batch_tab:
    st.subheader("Upload engineered product–store inputs")
    st.write(
        "Upload a UTF-8 CSV with the ten required columns. "
        "The maximum batch size is 5,000 records."
    )

    # Offer a schema template without inventing example input values.
    template = pd.DataFrame(columns=CONTRACT["feature_columns"])
    st.download_button(
        "Download empty CSV template",
        data=template.to_csv(index=False),
        file_name="superkart_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Prediction CSV", type=["csv"])

    if st.button("Predict batch", disabled=uploaded is None):
        # Clear any previous result before processing a new request.
        st.session_state.pop("batch_result", None)

        csv_bytes = uploaded.getvalue()

        with st.spinner("Processing batch..."):
            result = call_api(
                "/v1/predictbatch",
                files={"file": (uploaded.name, csv_bytes, "text/csv")},
            )

        if result is not None:
            inputs = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
            output = inputs.copy()
            output["Predicted_Revenue"] = result["predictions"]

            warning_map = {
                item["row"]: " ".join(item["messages"])
                for item in result.get("warnings", [])
            }
            output["Input_Warnings"] = [
                warning_map.get(index, "") for index in range(len(output))
            ]

            # Retain results so clicking Download does not erase them.
            st.session_state["batch_result"] = {
                "filename": uploaded.name,
                "output": output,
                "response": result,
            }

    if "batch_result" in st.session_state:
        saved = st.session_state["batch_result"]
        st.caption(f"Results from: {saved['filename']}")
        st.dataframe(
            saved["output"],
            hide_index=True,
            use_container_width=True,
        )
        show_warnings(saved["response"])

        st.download_button(
            "Download predictions and warnings",
            data=saved["output"].to_csv(index=False),
            file_name="superkart_predictions.csv",
            mime="text/csv",
        )
        st.caption(
            f"Model version: {saved['response']['model_version']}"
        )
