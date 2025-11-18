import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import time
import json

# === Global Color Settings ===
# Marsala color theme
MARSALA = "#955251"            # Main Marsala color
PRIMARY_COLOR = "#955251"      # Marsala
SECONDARY_COLOR = "#C08081"    # Light Marsala
SUCCESS_COLOR = "#6B8E23"      # Olive green
DANGER_COLOR = "#8B2E2E"       # Deep red
BACKGROUND_COLOR = "#FAFAFA"   # Light background
TEXT_COLOR = "#2E2E2E"         # Dark text



# Page configuration
st.set_page_config(
    page_title="Crypto Price Prediction Dashboard",
    page_icon="😻",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Apply base CSS
st.markdown(f"""
<style>
    body {{
        background-color: {BACKGROUND_COLOR};
        color: {MARSALA};
    }}
    .stMetric {{
        background-color: white;
        border: 1px solid #DDD;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }}
    .main-header {{
        color: {PRIMARY_COLOR};
    }}
</style>
""", unsafe_allow_html=True)

# Set Plotly defaults (affects all charts)
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.templates["plotly_white"]["layout"]["colorway"] = [
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, DANGER_COLOR, "#EDC948", "#B07AA1"
]

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #955251;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Paths
REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "final_data" / "crypto_only_related_data.csv"
METRICS_PATH = REPO_ROOT / "reports" / "metrics.json"

# API Configuration
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

# Sidebar
#st.sidebar.title("Configuration")
#st.sidebar.markdown("---")

# Main title
st.markdown('<div class="main-header"> Time Series Models on Cryptocurrency Data Set</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "EDA",
    "Model Metrics",
    "Prediction",
    "Model Comparison",
    "Analytics"
])

# ========================================
# TAB 1: EDA (Exploratory Data Analysis)
# ========================================
with tab1:
    st.header("Exploratory Data Analysis")

    # Data loading with caching
    @st.cache_data
    def load_crypto_data():
        """Load and cache cryptocurrency data"""
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        return None

    df = load_crypto_data()

    if df is not None:
        st.success(f"Data loaded: {len(df):,} rows, {len(df.columns)} columns")

        # Dataset Overview
        st.subheader("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            if 'ticker' in df.columns:
                st.metric("Cryptocurrencies", df['ticker'].nunique())
            else:
                st.metric("Cryptocurrencies", "N/A")
        with col3:
            if 'date' in df.columns:
                st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
            else:
                st.metric("Date Range", "N/A")
        with col4:
            st.metric("Features", len(df.columns))

        # Data Preview
        #st.subheader("Data Preview")
        #st.dataframe(df.head(20), use_container_width=True)

        # Column Selection for Analysis
        st.subheader("Price Analysis")

        if 'ticker' in df.columns:
            # Ticker selection
            available_tickers = df['ticker'].unique()
            selected_ticker = st.selectbox(
                "Select Cryptocurrency:",
                options=available_tickers,
                index=0 if len(available_tickers) > 0 else None
            )

            if selected_ticker:
                ticker_df = df[df['ticker'] == selected_ticker].copy()

                if 'date' in ticker_df.columns:
                    ticker_df = ticker_df.sort_values('date')

                # Price Time Series
                if 'close' in ticker_df.columns:
                    st.markdown("### Price Over Time")

                    fig = go.Figure()

                    # Close price
                    fig.add_trace(go.Scatter(
                        x=ticker_df['date'] if 'date' in ticker_df.columns else ticker_df.index,
                        y=ticker_df['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ))

                    # Add high/low if available
                    if 'high' in ticker_df.columns and 'low' in ticker_df.columns:
                        fig.add_trace(go.Scatter(
                            x=ticker_df['date'] if 'date' in ticker_df.columns else ticker_df.index,
                            y=ticker_df['high'],
                            mode='lines',
                            name='High',
                            line=dict(color='green', width=1, dash='dash'),
                            opacity=0.5
                        ))
                        fig.add_trace(go.Scatter(
                            x=ticker_df['date'] if 'date' in ticker_df.columns else ticker_df.index,
                            y=ticker_df['low'],
                            mode='lines',
                            name='Low',
                            line=dict(color='red', width=1, dash='dash'),
                            opacity=0.5
                        ))

                    fig.update_layout(
                        title=f"{selected_ticker} Price History",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified',
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Volume Analysis
                if 'volume' in ticker_df.columns:
                    st.markdown("### Trading Volume")

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=ticker_df['date'] if 'date' in ticker_df.columns else ticker_df.index,
                        y=ticker_df['volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ))

                    fig.update_layout(
                        title=f"{selected_ticker} Trading Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

        # Statistical Summary
        st.subheader("Statistical Summary")

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            selected_cols = st.multiselect(
                "Select features to analyze:",
                options=numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
            )

            if selected_cols:
                st.dataframe(df[selected_cols].describe(), use_container_width=True)

                # Distribution plots
                st.markdown("### Feature Distributions")

                num_plots = len(selected_cols)
                num_cols = 2
                num_rows = (num_plots + num_cols - 1) // num_cols

                for i in range(0, num_plots, num_cols):
                    cols = st.columns(num_cols)
                    for j, col in enumerate(cols):
                        if i + j < num_plots:
                            feature = selected_cols[i + j]
                            with col:
                                fig = px.histogram(
                                    df,
                                    x=feature,
                                    title=f"Distribution of {feature}",
                                    nbins=50
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)

        # Correlation Matrix
        st.subheader("Feature Correlations")

        if len(numeric_cols) > 1:
            # Select top features for correlation
            corr_features = st.multiselect(
                "Select features for correlation analysis:",
                options=numeric_cols,
                default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols
            )

            if len(corr_features) > 1:
                corr_matrix = df[corr_features].corr()

                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))

                fig.update_layout(
                    title="Feature Correlation Matrix",
                    height=max(400, len(corr_features) * 30),
                    xaxis_tickangle=-45
                )

                st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f" Data file not found at: {DATA_PATH}")
        st.info("Please ensure the data file exists or upload a CSV file below:")

        uploaded_file = st.file_uploader("Upload CSV Data", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            st.dataframe(df.head())

# ========================================
# TAB 2: MODEL METRICS
with tab2:
    st.header("Model Performance Metrics")

    # Load metrics
    @st.cache_data
    def load_metrics():
        """Load model metrics from JSON file"""
        if METRICS_PATH.exists():
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
        return None

    metrics = load_metrics()

    if metrics:
        st.success(" Metrics loaded successfully")

        # Overall Performance Comparison
        st.subheader("Model Performance Overview")

        # Extract MAE and MSE for each model
        model_names = []
        mae_values = []
        mse_values = []
        rmse_values = []

        for model, vals in metrics.items():
            if "MAE_original" in vals and "MSE_original" in vals:
                model_names.append(model)
                mae_values.append(vals["MAE_original"])
                mse_values.append(vals["MSE_original"])
                rmse_values.append(np.sqrt(vals["MSE_original"]))

        # Metrics Summary Cards
        col1, col2, col3 = st.columns(3)

        # Find best model by MAE
        best_mae_idx = np.argmin(mae_values)
        best_mse_idx = np.argmin(mse_values)

        with col1:
            st.metric(
                "Best Model (MAE)",
                model_names[best_mae_idx],
                f"MAE: {mae_values[best_mae_idx]:.2f}"
            )

        with col2:
            st.metric(
                "Best Model (MSE)",
                model_names[best_mse_idx],
                f"MSE: {mse_values[best_mse_idx]:.2f}"
            )

        with col3:
            if "ENSEMBLE" in metrics:
                st.metric(
                    "Ensemble Performance",
                    f"MAE: {metrics['ENSEMBLE']['MAE_original']:.2f}",
                    f"MSE: {metrics['ENSEMBLE']['MSE_original']:.2f}"
                )

        # Detailed Metrics Table
        st.subheader("Detailed Metrics Comparison")

        metrics_df = pd.DataFrame({
            'Model': model_names,
            'MAE': [f"{v:.4f}" for v in mae_values],
            'MSE': [f"{v:.4f}" for v in mse_values],
            'RMSE': [f"{v:.4f}" for v in rmse_values]
        })

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # MAE Comparison
            fig = go.Figure()

            colors = ['#FF6B6B' if name != 'ENSEMBLE' else '#4ECDC4' for name in model_names]

            fig.add_trace(go.Bar(
                x=model_names,
                y=mae_values,
                marker_color=colors,
                text=[f"{v:.2f}" for v in mae_values],
                textposition='auto',
                name='MAE'
            ))

            fig.update_layout(
                title="Mean Absolute Error (MAE) Comparison",
                xaxis_title="Model",
                yaxis_title="MAE",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # RMSE Comparison
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=model_names,
                y=rmse_values,
                marker_color=colors,
                text=[f"${v:.2f}" for v in rmse_values],
                textposition='auto',
                name='RMSE'
            ))

            fig.update_layout(
                title="Root Mean Squared Error (RMSE) Comparison",
                xaxis_title="Model",
                yaxis_title="RMSE",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # Radar Chart for Model Comparison
        st.subheader("Multi-Metric Model Comparison")

        # Normalize metrics to 0-100 scale (inverted - higher is better)
        mae_norm = 100 - ((np.array(mae_values) - min(mae_values)) / (max(mae_values) - min(mae_values)) * 100)
        rmse_norm = 100 - ((np.array(rmse_values) - min(rmse_values)) / (max(rmse_values) - min(rmse_values)) * 100)

        fig = go.Figure()

        for i, model in enumerate(model_names):
            fig.add_trace(go.Scatterpolar(
                r=[mae_norm[i], rmse_norm[i]],
                theta=['MAE Score', 'RMSE Score'],
                fill='toself',
                name=model
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Model Performance Scores (Higher is Better)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed Model Information
        st.subheader("Model Details")

        selected_model = st.selectbox("Select model for details:", model_names)

        if selected_model and selected_model in metrics:
            model_metrics = metrics[selected_model]

            st.json(model_metrics)

            # If ensemble, show base models
            if selected_model == "ENSEMBLE" and "base_models" in model_metrics:
                st.markdown("**Ensemble Components:**")
                for base_model in model_metrics["base_models"]:
                    st.write(f"- {base_model}")

    else:
        st.warning(f"Metrics file not found at: {METRICS_PATH}")
        st.info("Run the evaluation script to generate metrics:")
        st.code("python evaluation/evaluate_models.py", language="bash")

# ========================================
# TAB 3: PREDICTION (existing functionality)
# ========================================
with tab3:
    st.header("Make a Prediction")

    st.subheader("Input Options")
    input_method = st.radio(
        "Choose input method:",
        ["Random Sample", "Manual Input", "Upload CSV"]
    )

    if input_method == "Random Sample":
        st.info("Generate a random sample sequence for testing")
        n_features = st.number_input("Number of features", min_value=5, max_value=20, value=16)

        if st.button(" Generate Random Sample", type="primary"):
            sequence = []
            base_price = np.random.uniform(30000, 60000)

            for i in range(20):
                price_change = np.random.normal(0, 0.02)
                close = base_price * (1 + price_change)
                high = close * (1 + abs(np.random.normal(0, 0.01)))
                low = close * (1 - abs(np.random.normal(0, 0.01)))
                open_price = np.random.uniform(low, high)
                volume = np.random.uniform(1000000, 5000000)

                features = [
                    open_price, high, low, close, volume,
                    close * 0.98, close * 0.99, close,
                    close * 0.97, close * 0.98,
                    np.random.uniform(30, 70),
                    np.random.uniform(-100, 100),
                    np.random.uniform(-100, 100),
                    high * 1.02, close, low * 0.98
                ]

                sequence.append(features[:n_features])
                base_price = close

            st.session_state.sequence = sequence
            st.success(" Random sample generated!")

    elif input_method == "Manual Input":
        st.warning("Enter 20 timesteps with feature values separated by commas")
        sequence_text = st.text_area(
            "Paste sequence data (one timestep per line):",
            height=300,
            help="Format: Each line should contain comma-separated feature values"
        )

        if st.button("Parse Input", type="primary"):
            try:
                lines = [line.strip() for line in sequence_text.split('\n') if line.strip()]
                sequence = [[float(x.strip()) for x in line.split(',')] for line in lines]

                if len(sequence) != 20:
                    st.error(f"Expected 20 timesteps, got {len(sequence)}")
                else:
                    st.session_state.sequence = sequence
                    st.success(f"✅ Parsed {len(sequence)} timesteps successfully!")
            except Exception as e:
                st.error(f"Error parsing input: {str(e)}")

    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                if len(df) >= 20:
                    sequence = df.tail(20).values.tolist()
                    st.session_state.sequence = sequence
                    st.success(f"Loaded last 20 rows from CSV")
                else:
                    st.error(f"CSV must have at least 20 rows, got {len(df)}")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")

    # Display and predict
    if 'sequence' in st.session_state:
        st.markdown("---")
        st.subheader("Current Sequence")

        df_seq = pd.DataFrame(st.session_state.sequence)
        st.dataframe(df_seq, use_container_width=True)

        if st.button(" Make Prediction", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    start_time = time.time()

                    payload = {"sequence": st.session_state.sequence}
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=10
                    )

                    latency = (time.time() - start_time) * 1000

                    if response.status_code == 200:
                        result = response.json()

                        st.success(f" Prediction completed in {latency:.2f}ms")

                        # Display results
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Ensemble Prediction",
                                f"{result['prediction']:,.2f}",
                                help="Average prediction from all models"
                            )

                        with col2:
                            predictions = list(result['components'].values())
                            std_dev = np.std(predictions)
                            st.metric(
                                "Prediction Std Dev",
                                f"{std_dev:,.2f}",
                                help="Standard deviation across model predictions"
                            )

                        with col3:
                            confidence = max(0, 100 - (std_dev / result['prediction'] * 100))
                            st.metric(
                                "Confidence",
                                f"{confidence:.1f}%",
                                help="Based on model agreement"
                            )

                        # Model breakdown
                        st.subheader("Model Predictions")

                        fig = go.Figure()

                        models = list(result['components'].keys())
                        values = list(result['components'].values())

                        fig.add_trace(go.Bar(
                            x=models,
                            y=values,
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                            text=[f"{v:,.2f}" for v in values],
                            textposition='auto',
                        ))

                        fig.add_hline(
                            y=result['prediction'],
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"Ensemble: {result['prediction']:,.2f}"
                        )

                        fig.update_layout(
                            title="Model Predictions Comparison",
                            xaxis_title="Model",
                            yaxis_title="Predicted Price (USD)",
                            showlegend=False,
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Save to session
                        st.session_state.last_prediction = result
                        st.session_state.last_prediction_time = datetime.now()

                    else:
                        st.error(f"Prediction failed: {response.status_code}\n{response.text}")

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

# ========================================
# TAB 4: MODEL COMPARISON
# ========================================
with tab4:
    st.header("Model Performance Comparison")

    if 'last_prediction' in st.session_state:
        result = st.session_state.last_prediction

        col1, col2 = st.columns(2)

        with col1:
            # Radar chart
            models = list(result['components'].keys())
            values = list(result['components'].values())

            # Normalize values for radar chart
            min_val = min(values)
            max_val = max(values)
            normalized = [(v - min_val) / (max_val - min_val) * 100 if max_val != min_val else 50 for v in values]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=normalized,
                theta=models,
                fill='toself',
                name='Predictions'
            ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Model Predictions (Normalized)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box plot showing spread
            fig = go.Figure()

            for model, value in result['components'].items():
                fig.add_trace(go.Box(
                    y=[value],
                    name=model,
                    boxmean=True
                ))

            fig.update_layout(
                title="Prediction Distribution",
                yaxis_title="Predicted Price (USD)",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        # Detailed comparison table
        st.subheader("Detailed Model Comparison")

        ensemble_pred = result['prediction']
        comparison_data = []

        for model, pred in result['components'].items():
            deviation = pred - ensemble_pred
            deviation_pct = (deviation / ensemble_pred) * 100

            comparison_data.append({
                'Model': model,
                'Prediction': f"{pred:,.2f}",
                'Deviation from Ensemble': f"${deviation:,.2f}",
                'Deviation %': f"{deviation_pct:+.2f}%"
            })

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    else:
        st.info(" Make a prediction first to see model comparison")

# ========================================
# TAB 5: ANALYTICS
# ========================================
with tab5:
    st.header("Performance Analytics")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh Analytics", type="primary"):
            st.rerun()

    try:
        # Get performance metrics
        response = requests.get(f"{API_URL}/analytics/performance", timeout=5)

        if response.status_code == 200:
            metrics = response.json()

            if 'error' not in metrics:
                st.subheader("Model Performance Metrics")

                # Display metrics
                metric_cols = st.columns(4)

                model_names = ['ann', 'gru', 'lstm', 'transformer']

                for idx, model in enumerate(model_names):
                    with metric_cols[idx]:
                        if model in metrics:
                            avg_pred = metrics[model].get('avg_prediction', 0)
                            count = metrics[model].get('count', 0)

                            st.metric(
                                model.upper(),
                                f"{avg_pred:,.2f}",
                                f"{count} predictions"
                            )

                # Visualize average predictions
                if any(model in metrics for model in model_names):
                    fig = go.Figure()

                    for model in model_names:
                        if model in metrics:
                            fig.add_trace(go.Bar(
                                name=model.upper(),
                                x=['Average Prediction'],
                                y=[metrics[model].get('avg_prediction', 0)],
                            ))

                    fig.update_layout(
                        title="Average Predictions by Model",
                        yaxis_title="Price (USD)",
                        barmode='group',
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(metrics['error'])
        else:
            st.error("Failed to fetch analytics")

    except Exception as e:
        st.error(f"Error fetching analytics: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Cryptocurrency Price Prediction Dashboard | Powered by Ensemble ML Models</p>
    </div>
    """,
    unsafe_allow_html=True
)
