from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "netsentinel_autoencoder.keras"
PREPROCESSOR_PATH = ROOT / "models" / "netsentinel_preprocessing.joblib"
MAX_ROWS = 250_000

st.set_page_config(
    page_title="NetSentinel | Anomaly Observatory",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #e8f4ff;
      --muted: #91a6bc;
      --cyan: #39e7ff;
      --blue: #5487ff;
      --magenta: #d56bff;
      --green: #47f2bd;
      --red: #ff668f;
      --panel: rgba(14, 27, 46, .78);
      --line: rgba(111, 179, 255, .16);
    }
    [data-testid="stAppViewContainer"] {
      color: var(--ink);
      background:
        radial-gradient(ellipse at 76% 4%, rgba(43, 96, 180, .18), transparent 36%),
        radial-gradient(ellipse at 10% 24%, rgba(31, 200, 218, .08), transparent 30%),
        #07111f;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0b192a 0%, #091321 100%);
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li { color: #afc2d6; }
    .main .block-container { padding: 2.1rem 2.6rem 3rem; max-width: 1500px; }
    h1, h2, h3 { color: var(--ink) !important; letter-spacing: -.025em; }
    h1 { font-size: 2.5rem !important; font-weight: 740 !important; }
    h2 { font-size: 1.35rem !important; }
    p, label, [data-testid="stMarkdownContainer"] { color: #b9cadd; }
    .ns-brandline { display:flex; align-items:center; gap:.7rem; color:#93a9c0;
      font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; margin-bottom:.75rem; }
    .ns-dot { width:8px; height:8px; border-radius:50%; background:var(--green);
      box-shadow:0 0 13px var(--green); display:inline-block; }
    .ns-title-sub { color:#91a6bc; margin-top:-.6rem; margin-bottom:1.1rem; }
    .ns-banner { border:1px solid rgba(57,231,255,.22); border-left:3px solid var(--cyan);
      background:linear-gradient(100deg,rgba(23,61,91,.64),rgba(17,31,52,.72));
      border-radius:12px; padding:14px 18px; color:#c9e6f4; margin:.75rem 0 1.2rem; }
    .ns-panel { background:linear-gradient(145deg,rgba(18,35,57,.94),rgba(10,22,38,.92));
      border:1px solid var(--line); border-radius:16px; padding:18px 20px;
      box-shadow:0 12px 38px rgba(0,0,0,.22); }
    .ns-panel-title { color:#8ea7bd; text-transform:uppercase; letter-spacing:.13em;
      font-size:.69rem; font-weight:700; margin-bottom:.65rem; }
    .ns-status { color:var(--green); font-size:1.25rem; font-weight:750; letter-spacing:.06em; }
    .ns-threshold { color:#d8e7f6; font-size:1.65rem; font-weight:700; margin:.2rem 0; }
    .ns-caption { color:var(--muted); font-size:.82rem; line-height:1.5; }
    .ns-section { display:flex; align-items:center; gap:.65rem; color:#dcecff;
      font-size:1.08rem; font-weight:700; margin:1.45rem 0 .75rem; }
    .ns-section:before { content:""; width:4px; height:19px; border-radius:5px;
      background:linear-gradient(var(--cyan),var(--blue)); box-shadow:0 0 12px rgba(57,231,255,.45); }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(19,39,63,.93),rgba(11,25,43,.93));
      border:1px solid var(--line); border-radius:14px; padding:15px 18px; }
    div[data-testid="stMetricLabel"] { color:#8fa9c2; }
    div[data-testid="stMetricValue"] { color:#eff9ff; }
    div[data-testid="stFileUploaderDropzone"] { background:rgba(11,27,46,.72);
      border:1px dashed rgba(57,231,255,.46); border-radius:14px; }
    div[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--cyan); }
    .stButton > button, .stDownloadButton > button { border:1px solid rgba(57,231,255,.38);
      color:#dffaff; background:linear-gradient(100deg,rgba(27,110,153,.5),rgba(60,78,157,.48));
      border-radius:10px; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--cyan); color:white; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    .ns-profile-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:14px 0 12px; }
    .ns-profile-cell { background:rgba(7,18,32,.52); border:1px solid rgba(111,179,255,.12);
      border-radius:11px; padding:13px 14px; }
    .ns-profile-label { color:#8fa9c2; font-size:.65rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .ns-profile-value { color:#eaf7ff; font-size:1.35rem; font-weight:750; margin-top:4px; }
    .ns-profile-detail { color:#93a9bd; font-size:.73rem; margin-top:2px; }
    .ns-architecture { color:#b9d3e8; font-size:.83rem; border-top:1px solid var(--line); padding-top:12px; margin-top:8px; }
    .ns-empty { color:#92a9bd; border:1px solid var(--line); border-radius:14px;
      background:rgba(12,26,44,.68); padding:22px; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained model…")
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    bundle = joblib.load(PREPROCESSOR_PATH)
    preprocessor = bundle["preprocessor"]
    threshold = float(bundle["threshold"])
    expected_columns = list(preprocessor.feature_names_in_)
    return model, preprocessor, threshold, expected_columns


def score_rows(model, preprocessor, frame, feature_columns):
    encoded = preprocessor.transform(frame.loc[:, feature_columns])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded = np.asarray(encoded, dtype=np.float32)

    scores = []
    for start in range(0, len(encoded), 4096):
        batch = encoded[start : start + 4096]
        reconstructed = model.predict(batch, verbose=0)
        scores.extend(np.mean(np.square(batch - reconstructed), axis=1).tolist())
    return np.asarray(scores, dtype=np.float32)


def evaluate_labels(labels, predictions, scores):
    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "Attack precision": precision_score(labels, predictions, zero_division=0),
        "Attack recall": recall_score(labels, predictions, zero_division=0),
        "Attack F1": f1_score(labels, predictions, zero_division=0),
    }
    if len(np.unique(labels)) == 2:
        metrics["ROC-AUC"] = roc_auc_score(labels, scores)
        metrics["PR-AUC"] = average_precision_score(labels, scores)
    return metrics


def render_model_profile(model, preprocessor, feature_columns):
    counts = {name: 0 for name in ["numeric", "categorical"]}
    for name, _transformer, selected_columns in preprocessor.transformers_:
        if name in counts:
            counts[name] = len(selected_columns)
    encoded_width = int(model.input_shape[-1])
    latent_width = int(model.get_layer("latent_space").units)
    st.markdown(
        f"""
        <div class="ns-panel" style="min-height:305px;">
          <div class="ns-panel-title">Detection pipeline · Autoencoder</div>
          <div class="ns-profile-grid">
            <div class="ns-profile-cell"><div class="ns-profile-label">Flow inputs</div>
              <div class="ns-profile-value">{len(feature_columns)}</div><div class="ns-profile-detail">{counts['numeric']} numeric · {counts['categorical']} categorical</div></div>
            <div class="ns-profile-cell"><div class="ns-profile-label">Encoded vector</div>
              <div class="ns-profile-value">{encoded_width}</div><div class="ns-profile-detail">values after preprocessing</div></div>
            <div class="ns-profile-cell"><div class="ns-profile-label">Latent space</div>
              <div class="ns-profile-value">{latent_width}</div><div class="ns-profile-detail">compressed representation</div></div>
          </div>
          <div class="ns-architecture">Dense layers: <b>128 → 64 → {latent_width} → 64 → 128</b><br/>
          Trained on benign flows · Reconstruction error sets the anomaly score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_charts(results, threshold):
    score_frame = results[["anomaly_score", "model_result"]].copy()
    score_frame["log_score"] = np.log10(np.maximum(score_frame["anomaly_score"], 1e-8))
    threshold_log = float(np.log10(max(threshold, 1e-8)))
    colors = alt.Scale(
        domain=["Looks normal", "Flag for review"],
        range=["#43dfc2", "#ff668f"],
    )

    histogram = (
        alt.Chart(score_frame)
        .mark_bar(opacity=0.88, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("log_score:Q", bin=alt.Bin(maxbins=34), title="log₁₀ reconstruction error"),
            y=alt.Y("count():Q", title="Flows"),
            color=alt.Color("model_result:N", scale=colors, title="Decision"),
            tooltip=[alt.Tooltip("count():Q", title="Flows"), alt.Tooltip("model_result:N", title="Decision")],
        )
        .properties(height=270)
    )
    threshold_rule = (
        alt.Chart(pd.DataFrame({"cutoff": [threshold_log]}))
        .mark_rule(color="#f4c95d", strokeDash=[6, 4], strokeWidth=2)
        .encode(x="cutoff:Q")
    )
    st.altair_chart((histogram + threshold_rule).configure_view(strokeOpacity=0).configure_axis(
        labelColor="#9db2c7", titleColor="#c6d8ea", gridColor="#1e334b", domainColor="#35506d"
    ).configure_legend(labelColor="#bdcde0", titleColor="#d8e8f8"), width="stretch")

    distribution = results["model_result"].value_counts().rename_axis("decision").reset_index(name="flows")
    donut = (
        alt.Chart(distribution)
        .mark_arc(innerRadius=54, outerRadius=80, cornerRadius=4)
        .encode(
            theta=alt.Theta("flows:Q", stack=True),
            color=alt.Color("decision:N", scale=colors, title="Decision"),
            tooltip=[alt.Tooltip("decision:N", title="Decision"), alt.Tooltip("flows:Q", title="Flows")],
        )
        .properties(height=250, padding={"top": 18, "bottom": 22, "left": 18, "right": 18})
    )
    st.altair_chart(donut.configure_view(strokeOpacity=0).configure_legend(
        labelColor="#bdcde0", titleColor="#d8e8f8", orient="bottom"
    ), width="stretch")


def render_protocol_chart(results):
    if "proto" not in results.columns:
        return
    protocol_summary = (
        results.assign(proto=results["proto"].fillna("unknown").astype(str))
        .groupby(["proto", "model_result"], dropna=False)
        .size()
        .reset_index(name="flows")
    )
    top_protocols = results["proto"].fillna("unknown").astype(str).value_counts().head(10).index.tolist()
    protocol_summary = protocol_summary[protocol_summary["proto"].isin(top_protocols)]
    chart = (
        alt.Chart(protocol_summary)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("sum(flows):Q", title="Flows"),
            y=alt.Y("proto:N", sort="-x", title="Protocol"),
            color=alt.Color(
                "model_result:N",
                scale=alt.Scale(domain=["Looks normal", "Flag for review"], range=["#43dfc2", "#ff668f"]),
                title="Decision",
            ),
            tooltip=[alt.Tooltip("proto:N", title="Protocol"), alt.Tooltip("sum(flows):Q", title="Flows"), alt.Tooltip("model_result:N", title="Decision")],
        )
        .properties(height=310)
    )
    st.altair_chart(chart.configure_view(strokeOpacity=0).configure_axis(
        labelColor="#9db2c7", titleColor="#c6d8ea", gridColor="#1e334b", domainColor="#35506d"
    ).configure_legend(labelColor="#bdcde0", titleColor="#d8e8f8"), width="stretch")


st.markdown('<div class="ns-brandline"><span class="ns-dot"></span> NETSENTINEL&nbsp; / &nbsp;SECURITY OPERATIONS</div>', unsafe_allow_html=True)
st.title("Network anomaly observatory")
st.markdown('<div class="ns-title-sub">Deep learning analysis for network flow telemetry</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ns-banner"><b>OFFLINE ANALYSIS</b> &nbsp; Upload a flow CSV to score records. This dashboard does not capture or monitor live network traffic.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🛡️ NETSENTINEL")
    st.caption("AUTOENCODER · UNSW-NB15")
    st.markdown("---")
    st.markdown("#### Detection logic")
    st.write("The autoencoder learns patterns from benign flows. A high reconstruction error means a flow is unusual relative to that training data.")
    st.write("An alert is a review signal, not proof of an attack.")
    st.markdown("---")
    st.caption("MODEL INPUT · 42 flow features")

if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
    st.error("Model files are missing. Put the Keras model and preprocessing bundle in the `models` folder.")
    st.stop()

try:
    model, preprocessor, threshold, feature_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.info("Use the TensorFlow and scikit-learn versions listed in `requirements.txt`.")
    st.stop()

top_profile, top_status = st.columns([2.3, 1], gap="large")
with top_profile:
    render_model_profile(model, preprocessor, feature_columns)
with top_status:
    st.markdown(
        f"""
        <div class="ns-panel" style="min-height: 305px;">
          <div class="ns-panel-title">Model status</div>
          <div class="ns-status">● &nbsp; ONLINE</div>
          <div style="height:19px"></div>
          <div class="ns-panel-title">Alert threshold</div>
          <div class="ns-threshold">{threshold:.5f}</div>
          <div class="ns-caption">Reconstruction error above this level is flagged for review.</div>
          <div style="height:23px"></div>
          <div class="ns-panel-title">Processing mode</div>
          <div class="ns-caption">CSV batch scoring<br/>No live packet capture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="ns-section">Load flow telemetry</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload a network-flow CSV",
    type=["csv"],
    help="The CSV needs the same feature columns used during training. Optional label fields are only used for evaluation.",
    max_upload_size=100,
)

if uploaded is None:
    st.markdown(
        '<div class="ns-empty"><b>Awaiting a flow batch</b><br/>Use a CSV in UNSW-NB15 format. The model expects its 42 flow-feature columns; <code>id</code>, <code>label</code>, and <code>attack_cat</code> are not model inputs. Create a small demo file from <code>test_df</code> in the Colab notebook.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    data = pd.read_csv(uploaded, low_memory=False, nrows=MAX_ROWS + 1)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()
if len(data) > MAX_ROWS:
    st.error(f"This demo accepts up to {MAX_ROWS:,} rows per upload.")
    st.stop()

missing_columns = [column for column in feature_columns if column not in data.columns]
if missing_columns:
    st.error(f"This CSV is missing {len(missing_columns)} required feature columns: " + ", ".join(missing_columns))
    st.stop()

feature_data = data.loc[:, feature_columns]
if feature_data.isna().any().any():
    missing_counts = feature_data.isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0].to_dict()
    st.error(f"Some required features contain blank values: {columns_with_missing}")
    st.stop()

try:
    with st.spinner("Scoring network flows…"):
        anomaly_scores = score_rows(model, preprocessor, data, feature_columns)
except Exception as exc:
    st.error(f"Could not preprocess or score these rows: {exc}")
    st.stop()

predictions = (anomaly_scores > threshold).astype(np.int8)
results = data.copy()
results["anomaly_score"] = anomaly_scores
results["model_result"] = np.where(predictions == 1, "Flag for review", "Looks normal")
results["model_label"] = predictions
results = results.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

flagged_count = int(predictions.sum())
total_count = len(predictions)
mean_score = float(anomaly_scores.mean())

st.markdown('<div class="ns-section">Batch overview</div>', unsafe_allow_html=True)
metric_cols = st.columns(4)
metric_cols[0].metric("Flows analyzed", f"{total_count:,}")
metric_cols[1].metric("Flagged for review", f"{flagged_count:,}")
metric_cols[2].metric("Flag rate", f"{flagged_count / total_count:.1%}")
metric_cols[3].metric("Mean anomaly score", f"{mean_score:.5f}")

st.markdown('<div class="ns-section">Signal intelligence</div>', unsafe_allow_html=True)
chart_left, chart_right = st.columns([1.55, 1], gap="large")
with chart_left:
    st.markdown("#### Reconstruction score distribution")
    render_score_charts(results, threshold)
with chart_right:
    st.markdown("#### Flows by protocol")
    render_protocol_chart(results)

st.markdown('<div class="ns-section">Flagged flow queue</div>', unsafe_allow_html=True)
only_flagged = st.checkbox("Show flagged flows only", value=True)
display_results = results[results["model_label"] == 1] if only_flagged else results

display_columns = [column for column in ["id", "model_result", "anomaly_score"] if column in display_results.columns]
display_columns.extend(column for column in feature_columns[:8] if column in display_results.columns)
if "attack_cat" in display_results.columns:
    display_columns.append("attack_cat")
if "label" in display_results.columns:
    display_results = display_results.copy()
    display_results["reference_label"] = display_results["label"].map({0: "normal", 1: "attack"})
    display_columns.append("reference_label")

st.dataframe(
    display_results.loc[:, display_columns],
    hide_index=True,
    width="stretch",
    column_config={
        "anomaly_score": st.column_config.NumberColumn("Anomaly score", format="%.5f"),
        "model_result": st.column_config.TextColumn("Model result"),
        "reference_label": st.column_config.TextColumn("Known label · evaluation only"),
    },
)

download = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download scored CSV",
    data=download,
    file_name="netsentinel_scored_flows.csv",
    mime="text/csv",
)

if "label" in data.columns:
    labels = pd.to_numeric(data["label"], errors="coerce")
    valid = labels.notna() & labels.isin([0, 1])
    if valid.all() and labels.nunique() == 2:
        st.markdown('<div class="ns-section">Evaluation against known labels</div>', unsafe_allow_html=True)
        actual = labels.astype(int).to_numpy()
        evaluation = evaluate_labels(actual, predictions, anomaly_scores)
        eval_cols = st.columns(len(evaluation))
        for column, (name, value) in zip(eval_cols, evaluation.items()):
            column.metric(name, f"{value:.4f}")

        matrix = confusion_matrix(actual, predictions, labels=[0, 1])
        matrix_rows = [
            {"actual": actual_name, "predicted": predicted_name, "flows": int(matrix[i, j])}
            for i, actual_name in enumerate(["Normal", "Attack"])
            for j, predicted_name in enumerate(["Normal", "Attack"])
        ]
        matrix_data = pd.DataFrame(matrix_rows)
        heatmap = alt.Chart(matrix_data).mark_rect(cornerRadius=5).encode(
            x=alt.X("predicted:N", sort=["Normal", "Attack"], title="Predicted"),
            y=alt.Y("actual:N", sort=["Normal", "Attack"], title="Actual"),
            color=alt.Color("flows:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
            tooltip=["actual:N", "predicted:N", "flows:Q"],
        )
        labels_chart = alt.Chart(matrix_data).mark_text(color="#eff9ff", fontSize=20, fontWeight="bold").encode(
            x=alt.X("predicted:N", sort=["Normal", "Attack"]),
            y=alt.Y("actual:N", sort=["Normal", "Attack"]),
            text=alt.Text("flows:Q", format=","),
        )
        st.altair_chart((heatmap + labels_chart).properties(height=230).configure_view(strokeOpacity=0).configure_axis(
            labelColor="#b9cadd", titleColor="#dcecff", domainColor="#35506d"
        ), width="stretch")
    else:
        st.warning("A `label` column must contain both 0 (normal) and 1 (attack) for evaluation metrics. Labels are never model inputs.")

st.caption(
    "Anomaly score is reconstruction error, not attack probability. A high score means unusual relative to the benign training examples."
)
