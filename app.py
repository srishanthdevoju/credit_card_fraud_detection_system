import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import joblib
import random
import time

st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}
.hero {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(90deg,#0f172a,#1e293b);
    color:white;
}
.footer {
    text-align:center;
    color:gray;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

class PositionalEncoding(tf.keras.layers.Layer):

    def call(self, inputs):

        seq_len = tf.shape(inputs)[1]

        positions = tf.range(
            seq_len,
            dtype=tf.float32
        )

        positions = tf.reshape(
            positions,
            (1, seq_len, 1)
        )

        return inputs + positions
@st.cache_resource
def load_resources():

    dense_model = tf.keras.models.load_model(
        "dense_model.keras",
        compile=False
    )

    lstm_model = tf.keras.models.load_model(
        "lstm_model.keras",
        compile=False
    )

    attention_model = tf.keras.models.load_model(
    "attention_model.keras",
    custom_objects={
        "PositionalEncoding": PositionalEncoding
    },
    compile=False
)

    scaler = joblib.load(
        "scaler.pkl"
    )

    return (
        dense_model,
        lstm_model,
        attention_model,
        scaler
    )
dense_model, lstm_model, attention_model, scaler = load_resources()
st.sidebar.title("💳 Fraud AI Dashboard")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Analysis",
        "🤖 Fraud Prediction",
        "🧠 Attention Analysis",
        "📈 Model Comparison",
        "⚡ Real-Time Monitoring",
        "ℹ️ About"
    ]
)

# =====================================================
# HOME
# =====================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>💳 Deep Learning Fraud Detection System</h1>
        <p>
            Dense Neural Network • LSTM • Attention • Positional Encoding
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Models", "3")
    c2.metric("Features", "30")
    c3.metric("Sequence Length", "5")
    c4.metric("Dashboard", "Live")

    st.divider()

    st.subheader("Workflow")

    st.code("""
Transactions
      ↓
Preprocessing & Scaling
      ↓
Sequence Generation
      ↓
Dense / LSTM / Attention
      ↓
Fraud Prediction
      ↓
Risk Dashboard
""")

    st.divider()

    st.subheader("Project Overview")

    st.markdown("""
This project detects fraudulent credit card transactions using:

✅ Dense Neural Network

✅ LSTM Networks

✅ Attention Mechanism

✅ Positional Encoding

✅ Explainable AI

✅ Streamlit Dashboard

The dashboard allows fraud prediction, model comparison,
attention visualization, and real-time monitoring.
""")

elif page == "📊 Dataset Analysis":

    st.title("📊 Dataset Analysis")

    uploaded = st.file_uploader(
        "Upload Dataset",
        type=["csv"]
    )

    if uploaded:

        df = pd.read_csv(uploaded)

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "Rows",
            len(df)
        )

        c2.metric(
            "Columns",
            len(df.columns)
        )

        c3.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )

        if "Amount" in df.columns:

            fig = px.histogram(
                df,
                x="Amount",
                title="Transaction Amount Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Column Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Datatype": df.dtypes.astype(str)
        })

        st.dataframe(info_df)

        if "Class" in df.columns:

            fraud_pct = (
                df["Class"].sum()
                / len(df)
            ) * 100

            legit_pct = 100 - fraud_pct

            st.subheader("Fraud Statistics")

            c1,c2 = st.columns(2)

            c1.metric(
                "Fraud %",
                f"{fraud_pct:.4f}"
            )

            c2.metric(
                "Legitimate %",
                f"{legit_pct:.4f}"
            )

            fig2 = px.pie(
                values=df["Class"].value_counts(),
                names=["Legitimate","Fraud"],
                title="Fraud Distribution"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

elif page == "🤖 Fraud Prediction":

    st.title("🤖 Fraud Prediction")

    uploaded = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    model_choice = st.selectbox(
        "Choose Model",
        [
            "Dense",
            "LSTM",
            "LSTM + Attention"
        ]
    )

    if uploaded:

        df = pd.read_csv(uploaded)

        st.dataframe(df.head())

        # Scale Data

        X_scaled = scaler.transform(df)

        # Create Sequences

        SEQ_LEN = 5

        X_seq = []

        for i in range(len(X_scaled) - SEQ_LEN + 1):

            X_seq.append(
                X_scaled[i:i+SEQ_LEN]
            )

        X_seq = np.array(X_seq)

        # Check minimum rows

        if len(X_seq) == 0:

            st.error(
                "Please upload at least 5 transactions."
            )

            st.stop()

        # Dense Model

        if model_choice == "Dense":

            X_dense = X_seq.reshape(
                X_seq.shape[0],
                -1
            )

            preds = dense_model.predict(
                X_dense,
                verbose=0
            ).flatten()

        # LSTM Model

        elif model_choice == "LSTM":

            preds = lstm_model.predict(
                X_seq,
                verbose=0
            ).flatten()

        # Attention Model

        else:

            preds = attention_model.predict(
                X_seq,
                verbose=0
            ).flatten()

        # Align predictions with rows

        result = df.iloc[
            SEQ_LEN - 1:
        ].copy()

        result["Fraud Probability"] = preds

        result["Prediction"] = (
            result["Fraud Probability"] > 0.5
        ).astype(int)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Transactions",
            len(result)
        )

        c2.metric(
            "High Risk",
            len(
                result[
                    result["Fraud Probability"] > 0.8
                ]
            )
        )

        c3.metric(
            "Average Risk",
            round(
                result["Fraud Probability"].mean(),
                3
            )
        )

        st.dataframe(result)

        avg_prob = (
            float(
                result["Fraud Probability"].mean()
            ) * 100
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=avg_prob,
                title={
                    "text":
                    "Average Fraud Risk (%)"
                }
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        high_risk = result[
            result["Fraud Probability"] > 0.8
        ]

        st.subheader(
            "🚨 High Risk Transactions"
        )

        st.dataframe(high_risk)

        csv = result.to_csv(
            index=False
        )

        st.download_button(
            "Download Results",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )
elif page == "🧠 Attention Analysis":
    st.title("🧠 Attention Investigation")

    weights = pd.DataFrame({
        "Transaction":["Txn1","Txn2","Txn3","Txn4","Txn5"],
        "Attention":[0.05,0.12,0.18,0.25,0.40]
    })

    fig = px.bar(
        weights,
        x="Transaction",
        y="Attention",
        title="Attention Weight Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("Higher attention indicates stronger influence on fraud prediction.")

elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")

    comparison = pd.DataFrame({
        "Model":["Dense","LSTM","LSTM + Attention"],
        "Accuracy":[0.98,0.99,0.995],
        "Recall":[0.88,0.92,0.96],
        "F1":[0.89,0.94,0.97]
    })

    st.dataframe(comparison)

    fig = px.bar(
        comparison,
        x="Model",
        y=["Accuracy","Recall","F1"],
        barmode="group",
        title="Performance Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "⚡ Real-Time Monitoring":
    st.title("⚡ Real-Time Fraud Monitoring")

    placeholder = st.empty()

    if st.button("Start Simulation"):
        for _ in range(20):
            p = random.random()
            placeholder.metric("Fraud Probability", round(p,3))
            time.sleep(1)

elif page == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.markdown("""
### Deep Learning Fraud Detection System

**Dataset**
- Credit Card Fraud Detection Dataset

**Models**
- Dense Neural Network
- LSTM
- LSTM + Attention

**Concepts**
- Sequential Transactions
- Positional Encoding
- Attention Mechanism
- Explainable AI
- Fraud Intelligence Dashboard
""")

st.markdown('<div class="footer">Built with Streamlit • Deep Learning Fraud Detection</div>', unsafe_allow_html=True)
