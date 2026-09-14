import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="🛒", layout="centered")

st.title("🛒 SuperKart Sales Predictor")
st.caption("Enter product + store details → get predicted sales total.")

# Backend URL (set this in Hugging Face Space Secrets / env vars)
# Example: https://<your-backend-space>.hf.space
BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://172.18.0.1:7860")
PREDICT_URL = f"{BACKEND_BASE_URL}/v1/predict"
PREDICT_BATCH_URL = f"{BACKEND_BASE_URL}/v1/predictbatch"

with st.sidebar:
    st.subheader("Backend")
    st.write("Using:", PREDICT_URL)
    st.caption("Tip: set `BACKEND_BASE_URL` in Space secrets for production.")

# ---- Input form ----
with st.form("predict_form"):
    st.subheader("Product Details")

    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.0, step=0.1)
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0, step=1.0)
    with col2:
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_type = st.selectbox(
            "Product Type",
            [
                "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
                "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Soft Drinks",
                "Meat", "Breads", "Hard Drinks", "Starchy Foods", "Others", "Seafood",
                "Breakfast"
            ],
        )

    st.subheader("Store Details")
    col3, col4 = st.columns(2)
    with col3:
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_city_tier = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    with col4:
        store_type = st.selectbox(
            "Store Type",
            ["Grocery Store", "Supermarket Type 1", "Supermarket Type 2", "Supermarket Type 3"]
        )
        store_age = st.number_input("Store Age", min_value=0, value=10, step=1)

    submitted = st.form_submit_button("Predict Sales")

# ---- Call backend ----
if submitted:
    payload = {
        "Product_Weight": float(product_weight),
        "Product_Sugar_Content": product_sugar,
        "Product_Type": product_type,
        "Product_MRP": float(product_mrp),
        "Store_Size": store_size,
        "Store_Location_City_Type": store_city_tier,
        "Store_Type": store_type,
        "Store_Age": int(store_age),
    }

    with st.spinner("Calling prediction API..."):
        try:
            resp = requests.post(PREDICT_URL, json=payload, timeout=25)

            if resp.status_code != 200:
                st.error(f"API Error ({resp.status_code})")
                st.code(resp.text)
            else:
                result = resp.json()
                pred = result.get("prediction", None)
                if pred is None:
                    st.error("No `prediction` field returned by API.")
                    st.json(result)
                else:
                    st.success("Prediction received ✅")
                    st.metric("Predicted Sales Total", f"{pred:,.2f}")

                    with st.expander("Request JSON sent to API"):
                        st.json(payload)

        except requests.exceptions.RequestException as e:
            st.error("Failed to reach backend API.")
            st.write(str(e))

# Section for batch prediction
# ---- Batch Prediction ----
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file for batch prediction",
    type=["csv"]
)

if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):

        try:
            # Read CSV once for displaying later
            uploaded_file.seek(0)
            input_df = pd.read_csv(uploaded_file)

            # Reset file pointer before sending to backend
            uploaded_file.seek(0)

            response = requests.post(
                PREDICT_BATCH_URL,
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv"
                    )
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                st.success(
                    f"Batch predictions completed — {result['n_rows']} rows"
                )

                preds = pd.DataFrame(result["predictions"])

                output = pd.concat(
                    [
                        input_df.reset_index(drop=True),
                        preds["predicted_sales_total"].reset_index(drop=True)
                    ],
                    axis=1
                )

                st.dataframe(output)

            else:
                st.error(f"API error ({response.status_code})")
                st.code(response.text)

        except requests.exceptions.RequestException as e:
            st.error("Failed to reach backend API.")
            st.write(str(e))

        except Exception as e:
            st.error("Batch prediction failed.")
            st.write(str(e))
