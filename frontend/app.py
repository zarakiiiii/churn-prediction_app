import streamlit as st
import pandas as pd
import requests

st.title("Customer Churn Prediction (Batch - CSV Upload)")

#uplaoding csv file
uploaded_file = st.file_uploader("Upload CSV", type = ["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df)

    required_columns = [
        "purchase_count",
        "customer_state",
        "last_review_score",
        "main_product_category",
        "last_product_category"
    ]

    if all(col in df.columns for col in required_columns):
        if st.button("Predict Churn"):
            predictions = []

            for _, row in df.iterrows():
                input_data = {
                    "purchase_count": row["purchase_count"],
                    "customer_state": row["customer_state"],
                    "last_review_score": row["last_review_score"],
                    "main_product_category": row["main_product_category"],
                    "last_product_category": row["last_product_category"]
                }

                try:
                    response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
                    prediction = response.json().get("prediction")
                except Exception as e:
                    prediction = f"Error: {e}"
                predictions.append(prediction)

            df["prediction"] = predictions

            st.subheader("Predictions")
            st.write(df)




        else:
            st.error(f"CSV must contain columns: {required_columns}")

import streamlit as st
import pandas as pd

# Encoded sample input
sample_df = pd.DataFrame({
    "purchase_count": [3, 7],
    "customer_state": [1.0, 2.0],              # encoded float values (e.g., CA=1.0, TX=2.0)
    "last_review_score": [4.5, 3.2],
    "main_product_category": [0.0, 1.0],       # e.g., electronics=0.0, clothing=1.0
    "last_product_category": [2.0, 3.0]        # e.g., mobile=2.0, shoes=3.0
})

# Encode as CSV
sample_csv = sample_df.to_csv(index=False).encode('utf-8')

# Download button for testing
st.download_button(
    label="📥 Download Encoded Sample CSV (for Testing)",
    data=sample_csv,
    file_name="sample_input_encoded.csv",
    mime="text/csv"
)

