import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageOps

# Set Streamlit config
st.set_page_config(page_title="ML Model Comparison Dashboard", layout="wide")

st.markdown("""
    <style>
        /* Main background */
        .main {
            background-color: #1f1f1f;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Header */
        .custom-header {
            margin-bottom: 25px;
            padding: 20px;
            border-radius: 12px;
            border: 3px solid #4A90E2;
            box-shadow: 0 0 15px rgba(0, 128, 255, 0.3);
            background: linear-gradient(135deg, #1f1f1f, #4A90E2);
            text-align: center;
        }

        .custom-header h1 {
            color: white;
            font-size: 42px;
            font-weight: 900;
        }

        .custom-header h4 {
            color: #f0f0f0;
            font-size: 20px;
            font-weight: 500;
        }

        /* Customize default widget styles */
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
        }

        .stSelectbox, .stTextInput, .stNumberInput {
            background-color: #2a2a2a;
            color: #ffffff;
        }

        /* Chart background fix if needed */
        .css-1v0mbdj {
            background-color: #1f1f1f !important;
        }
    </style>
""", unsafe_allow_html=True)




# Toggle for theme
dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True)

# Theme colors
if dark_mode:
    bg_color = "#0e1117"
    text_color = "#ffffff"
    card_color = "#222222"
    gradient = "YlGnBu"
    border_color = "#1f4e79"
else:
    bg_color = "#ffffff"
    text_color = "#000000"
    card_color = "#f1f3f6"
    gradient = "Blues"
    border_color = "#005792"

# Apply background styling
st.markdown(f"""
    <style>
        html, body, [class*="css"] {{
            background-color: {bg_color};
            color: {text_color};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {card_color};
        }}
        .stMetricValue {{
            color: {text_color} !important;
        }}
        .stDataFrame, .stPlotlyChart, .stExpander {{
            background-color: {card_color} !important;
            color: {text_color};
        }}
    </style>
""", unsafe_allow_html=True)

# Sidebar Logo (Rounded Image)
logo = Image.open("logo.png").resize((180, 180))
mask = Image.new("L", logo.size, 0)
draw = ImageDraw.Draw(mask)
draw.ellipse((0, 0, logo.size[0], logo.size[1]), fill=255)
logo = ImageOps.fit(logo, mask.size, centering=(0.5, 0.5))
logo.putalpha(mask)
st.sidebar.image(logo, use_container_width=True)

st.markdown("""
    <style>
        .custom-header {
            margin-bottom: 25px;
            padding: 20px;
            border-radius: 12px;
            border: 3px solid #4A90E2;
            box-shadow: 0 0 15px rgba(0, 128, 255, 0.3);
            background: linear-gradient(135deg, #1f1f1f, #4A90E2);
            text-align: center;
        }
        .custom-header h1 {
            color: white;
            font-size: 42px;
            font-weight: 900;
            font-family: "Segoe UI", sans-serif;
            margin-bottom: 10px;
        }
        .custom-header h4 {
            color: white;
            font-size: 20px;
            font-weight: 500;
            font-family: "Segoe UI", sans-serif;
        }
    </style>

    <div class="custom-header">
        <h1> ML Model Comparison Dashboard</h1>
        <h4>Compare the performance of different classification algorithms at a glance</h4>
    </div>
""", unsafe_allow_html=True)



# Load models
logistic_model = joblib.load('logistic_model.pkl')
dt_model = joblib.load('dt_model.pkl')
rf_model = joblib.load('rf_model.pkl')
knn_model = joblib.load('knn_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

# Load test data
x_test = joblib.load('x_test.pkl')
y_test = joblib.load('y_test.pkl').astype(int)

# Predictions
logistic_pred = logistic_model.predict(x_test).astype(int)
dt_pred = dt_model.predict(x_test).astype(int)
rf_pred = rf_model.predict(x_test).astype(int)
knn_pred = knn_model.predict(x_test).astype(int)
xgb_pred = xgb_model.predict(x_test).astype(int)

# Evaluation
def evaluate_model(name, y_pred):
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "model": name,
        "accuracy": acc,
        "precision": report['1']['precision'],
        "recall": report['1']['recall'],
        "f1": report['1']['f1-score'],
        "report": report,
        "confusion": confusion_matrix(y_test, y_pred)
    }

metrics = {
    'Logistic Regression': evaluate_model("Logistic Regression", logistic_pred),
    'Decision Tree': evaluate_model("Decision Tree", dt_pred),
    'Random Forest': evaluate_model("Random Forest", rf_pred),
    'KNN': evaluate_model("KNN", knn_pred),
    'XGBoost': evaluate_model("XGBoost", xgb_pred)
}

# Sidebar selection
selected_model = st.sidebar.selectbox("🔍 Select a Model to Analyze", list(metrics.keys()))
data = metrics[selected_model]

st.markdown("<br><br>", unsafe_allow_html=True)


# KPIs
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{data['accuracy']:.2f}")
col2.metric("Precision (Class 1)", f"{data['precision']:.2f}")
col3.metric("Recall (Class 1)", f"{data['recall']:.2f}")
col4.metric("F1-Score (Class 1)", f"{data['f1']:.2f}")

# Custom style for expander and inner box
st.markdown("""
<style>
    summary {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #FFD700 !important;
    }
    .custom-insight-box {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #444;
    }
    .custom-insight-box li {
        border-bottom: 1px solid #333;
        padding: 10px 0;
        list-style-type: none;
    }
    .custom-insight-box li:last-child {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

with st.expander("❓ Why Precision / Recall / F1-Score Might Be Low?"):
    st.markdown(f"""
<div class="custom-insight-box">
<ul>
  <li>✅ <strong>Imbalance in Data</strong>: The dataset may be skewed toward the majority class (class <strong>'0'</strong>), making the model less sensitive to the minority class (<strong>'1'</strong>).</li>

  <li>⚖️ <strong>Accuracy ≠ Performance</strong>: A high accuracy can be misleading when the model predicts the majority class more often, ignoring the minority class.</li>

  <li>🔄 <strong>Limited Model Learning</strong>: The model might struggle to learn distinct patterns for class <strong>'1'</strong> due to <strong>insufficient or noisy feature representation</strong>, leading to lower precision, recall, and F1-score.</li>

  <li>🌐 <strong>Real-world Complexity</strong>: Even after balancing techniques, factors like overlapping features, noise in data, or small dataset size can impact these scores.</li>

  <li>🎯 <strong>Focus on Business Goal</strong>: If class <strong>'1'</strong> represents a critical case (like fraud, disease, etc.), improving <strong>Recall</strong> and <strong>F1-score</strong> is more important than Accuracy.</li>
</ul>
</div>
""", unsafe_allow_html=True)



# Confusion Matrix
st.markdown("### 🧩 Confusion Matrix")
fig, ax = plt.subplots(figsize=(5, 4))  # Adjust width and height as needed
sns.heatmap(data['confusion'], annot=True, fmt='d', cmap=gradient, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - {selected_model}')
st.pyplot(fig)

# Classification Report
st.markdown("### 📋 Detailed Classification Report")
report_df = pd.DataFrame(data['report']).transpose()
st.dataframe(report_df.style.background_gradient(cmap=gradient).format("{:.2f}"))

# Model Comparison Chart
st.markdown("### 📈 Model Comparison on Metrics")
comparison_df = pd.DataFrame({
    "Model": [m['model'] for m in metrics.values()],
    "Accuracy": [m['accuracy'] for m in metrics.values()],
    "Precision": [m['precision'] for m in metrics.values()],
    "Recall": [m['recall'] for m in metrics.values()],
    "F1 Score": [m['f1'] for m in metrics.values()]
})

chart_type = st.selectbox("Choose Chart Type", ["Bar", "Radar"])

if chart_type == "Bar":
    fig = px.bar(comparison_df.melt(id_vars=["Model"], var_name="Metric"),
                 x="Model", y="value", color="Metric", barmode="group",
                 title="Model Performance Comparison",
                 color_discrete_sequence=px.colors.qualitative.Dark24 if dark_mode else px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)
else:
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    fig = go.Figure()
    for i, row in comparison_df.iterrows():
        fig.add_trace(go.Scatterpolar(r=[row[m] for m in categories],
                                      theta=categories,
                                      fill='toself',
                                      name=row['Model']))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                      title="Radar Chart Comparison")
    st.plotly_chart(fig, use_container_width=True)



from sklearn.metrics import roc_curve, auc, precision_recall_curve
import plotly.graph_objs as go

# Function to plot ROC and PR curves with explanations
def plot_roc_pr(y_true, y_score, model_name):
    st.markdown("### 📉 ROC & Precision-Recall Curve")

    col1, col2 = st.columns(2)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig1.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig1.update_layout(title=f"ROC Curve - {model_name} (AUC = {roc_auc:.2f})",
                       xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate",
                       plot_bgcolor=bg_color,
                       paper_bgcolor=bg_color,
                       font=dict(color=text_color))
    col1.plotly_chart(fig1, use_container_width=True)
    with col1:
        st.markdown(f"**Interpretation:** A ROC curve closer to the top-left indicates better performance. AUC = {roc_auc:.2f} means the model has a strong ability to distinguish between classes.")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'))
    fig2.update_layout(title=f"Precision-Recall Curve - {model_name}",
                       xaxis_title="Recall",
                       yaxis_title="Precision",
                       plot_bgcolor=bg_color,
                       paper_bgcolor=bg_color,
                       font=dict(color=text_color))
    col2.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("**Interpretation:** This curve helps understand how well the model balances precision and recall. A curve that stays high indicates better performance for detecting the positive class.")

# Dictionary mapping selected model to prediction probabilities
model_probs = {
    'Logistic Regression': logistic_model.predict_proba(x_test)[:, 1],
    'Decision Tree': dt_model.predict_proba(x_test)[:, 1],
    'Random Forest': rf_model.predict_proba(x_test)[:, 1],
    'KNN': knn_model.predict_proba(x_test)[:, 1],
    'XGBoost': xgb_model.predict_proba(x_test)[:, 1]
}

# Call the function for the selected model
plot_roc_pr(y_test, model_probs[selected_model], selected_model)

st.markdown("<br><br>", unsafe_allow_html=True) # creating vertical space


st.markdown("""
<div style='background-color:#1E1E1E; padding: 25px; border-radius: 15px; border: 2px solid #00F5D4;'>

<h2 style='text-align:center; color:#00F5D4;'>💡 Final Insight</h2>

<ul style="color:#ffffff; font-size:18px; line-height:1.8; list-style:none; padding-left:0;">
  <li><b>🎯 Random Forest</b> stands out with the highest accuracy (0.87) and precision (0.64) for class 1, making it the best model for this dataset despite a lower recall.</li>
  <hr style="border: 0; border-top: 1px solid #444;" />
  <li><b>📈 Logistic Regression</b> offers a balanced performance with an F1-score of 0.42 — slightly better than Random Forest in recall, but with lower precision.</li>
  <hr style="border: 0; border-top: 1px solid #444;" />
  <li><b>🌲 Decision Tree</b> underperforms with the lowest precision, recall, and F1-score (all at 0.30), suggesting overfitting or poor generalization.</li>
  <hr style="border: 0; border-top: 1px solid #444;" />
  <li><b>📊 K-Nearest Neighbors</b> shows consistent but modest performance across all metrics, indicating it's not ideal for imbalanced datasets like this one.</li>
  <hr style="border: 0; border-top: 1px solid #444;" />
  <li><b>⚡ XGBoost</b> provides a strong accuracy of 0.85 and decent precision (0.50), but its recall (0.30) suggests it misses many positives, limiting real-world reliability.</li>
  <hr style="border: 0; border-top: 1px solid #444;" />
  <li>✅ <b>Conclusion:</b> <u>Random Forest</u> is the most suitable model for this dataset when prioritizing high precision, while <u>Logistic Regression</u> gives a better balance between precision and recall.</li>
</ul>

</div>
""", unsafe_allow_html=True)

