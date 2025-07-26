import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and feature column order
model = pickle.load(open('../models/logreg_model.pkl', 'rb'))
feature_columns = pickle.load(open('../models/feature_columns.pkl', 'rb'))

# ✅ Updated page config with icon
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.markdown("Upload customer data (CSV) to predict who is likely to churn.")

# File uploader
uploaded_file = st.file_uploader("📁 Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load uploaded data
        input_df = pd.read_csv(uploaded_file)
        st.write("### 👁️ Preview of uploaded data:")
        st.dataframe(input_df.head(10), use_container_width=True)  # ✅ show 10 entries

        # Preprocess: one-hot encode
        input_processed = pd.get_dummies(input_df)

        # Add missing columns and ensure correct order
        for col in feature_columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[feature_columns]

        # Predict
        preds = model.predict(input_processed)
        probs = model.predict_proba(input_processed)[:, 1]

        # Combine results with original data
        result_df = input_df.copy()
        result_df['Churn_Prediction'] = preds
        result_df['Churn_Probability'] = (probs * 100).round(2).astype(str) + '%'  # ✅ formatted %

        # Display predictions
        st.write("### 📊 Predictions:")
        st.dataframe(result_df[['Churn_Prediction', 'Churn_Probability']], use_container_width=True)  # ✅ full display
        st.success("✅ Predictions generated successfully!")  # ✅ success msg

        # Download button
        csv_download = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download full predictions",
            data=csv_download,
            file_name='churn_predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

else:
    st.info("📂 Awaiting CSV upload.")
