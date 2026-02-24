import io
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Student Marks Dashboard",layout="wide",
                   initial_sidebar_state="expanded")
warnings.filterwarnings("ignore")
matplotlib.use("Agg")


# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0a0e1a; color: #e2e8f0; }
  section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2d45; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1f2d45;
    border-radius: 12px;
    padding: 16px;}
            
  [data-testid="stMetricValue"] { color: #00d4ff; font-size: 1.8rem; font-weight: 700; }
  [data-testid="stMetricLabel"] { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
  [data-testid="stMetricDelta"] { font-size: 0.82rem; }

  /* Dataframe */
  .stDataFrame { border-radius: 10px; border: 1px solid #1f2d45; }

  /* Selectbox, file uploader */
  .stSelectbox > div > div, .stFileUploader > div { background: #111827 !important; border-color: #1f2d45 !important; color: #e2e8f0 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 8px; padding: 8px 18px; font-weight: 600; font-size: 0.88rem; }
  .stTabs [aria-selected="true"] { background: #1f2d45 !important; color: #00d4ff !important; }

  /* Divider */
  hr { border-color: #1f2d45; }

  /* Pill badges */
  .pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
  }
  .pill-green  { background: rgba(16,185,129,.15); color: #10b981; border: 1px solid rgba(16,185,129,.3); }
  .pill-red    { background: rgba(239,68,68,.15);  color: #ef4444; border: 1px solid rgba(239,68,68,.3); }
  .pill-blue   { background: rgba(0,212,255,.12);  color: #00d4ff; border: 1px solid rgba(0,212,255,.3); }
  .pill-amber  { background: rgba(245,158,11,.12); color: #f59e0b; border: 1px solid rgba(245,158,11,.3); }

  /* Section headers */
  h1, h2, h3 { color: #e2e8f0 !important; }
  p, label, span { color: #cbd5e1; }
  .stMarkdown p { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

#Color palette
DARK_BG  = "#0a0e1a"
SURFACE  = "#111827"
SURFACE2 = "#1a2235"
BORDER   = "#1f2d45"
ACCENT   = "#00d4ff"
PURPLE   = "#7c3aed"
AMBER    = "#f59e0b"
GREEN    = "#10b981"
RED      = "#ef4444"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"

EXAM_COLS = ["CIA-1", "Mid-sem", "CIA-3"]
MAX_MARKS = {"CIA-1": 20, "Mid-sem": 50, "CIA-3": 20}

def fig_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(axis="y", color=BORDER, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def clean_percent(val):
    """Convert '78.40%' or 78.40 to float 78.40"""
    if isinstance(val, str):
        val = val.replace("%", "").strip()
    try:
        v = float(val)
        # if stored as decimal fraction (e.g. 0.78) convert to %
        return v if v > 1 else v * 100
    except Exception:
        return np.nan


def load_csv(file_obj) -> pd.DataFrame:
    df = pd.read_csv(file_obj)
    df.dropna(subset=["Name"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Attendence"] = df["Attendence"].apply(clean_percent)
    for col in EXAM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

import seaborn as sns
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with scaled columns (all out of 100)."""
    out = df.copy()
    out["att_sc"]  = out["Attendence"]
    out["cia1_sc"] = (out["CIA-1"]   / 20) * 100
    out["mid_sc"]  = (out["Mid-sem"] / 50) * 100
    out["cia3_sc"] = (out["CIA-3"]   / 20) * 100
    return out


#===========================MODEL TRAINING & PREDICTION==============================
@st.cache_resource
def train_model(training_file_bytes: bytes):
    df_train = pd.read_csv(io.BytesIO(training_file_bytes))
    df_train.dropna(subset=["Name"], inplace=True)
    df_train["Attendence"] = df_train["Attendence"].apply(clean_percent)
    for col in EXAM_COLS + ["finals"]:
        if col in df_train.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors="coerce")
    df_train.dropna(subset=EXAM_COLS + ["finals"], inplace=True)

    df_train = scale_features(df_train)
    X= df_train[["att_sc", "cia1_sc", "mid_sc", "cia3_sc"]]
    y= df_train["finals"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)

    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "R2": round(r2_score(y_test, y_pred), 2),
        "n_train":len(X_train),
        "n_test" :len(X_test),
        "coefs": dict(zip(["Attendance", "CIA-1", "Mid-Sem", "CIA-3"], model.coef_))}
    return model, metrics, df_train


#================================Charts==================================
def chart_class_averages(df: pd.DataFrame):
    avgs = [df[c].mean() for c in EXAM_COLS]
    maxm = [MAX_MARKS[c] for c in EXAM_COLS]
    colors = [ACCENT, PURPLE, AMBER]
    labels = ["CIA-1  (/20)", "Mid-Sem  (/50)", "CIA-3  (/20)"]

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=DARK_BG)
    x= np.arange(len(EXAM_COLS))
    w= 0.35
    bars1 = ax.bar(x - w/2, avgs, width=w, color=colors, alpha=0.85, edgecolor=colors, linewidth=1.2)
    ax.bar(x + w/2, maxm, width=w, color=colors, alpha=0.18, edgecolor=colors, linewidth=1, linestyle="--")
    for bar, val in zip(bars1, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25,
                f"{val:.1f}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
    fig_style(ax, "Class Average per Exam", ylabel="Marks")
    ax.set_xticks(x); ax.set_xticklabels(labels, color=TEXT, fontsize=9)
    leg = ax.legend(
        [mpatches.Patch(color=ACCENT, alpha=0.85), mpatches.Patch(color=MUTED, alpha=0.4)],
        ["Class Average", "Max Marks"],
        facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    fig.tight_layout(pad=1.5)
    return fig


def chart_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), facecolor=DARK_BG)
    pairs = [("CIA-1", ACCENT), ("Mid-sem", PURPLE), ("CIA-3", AMBER)]
    for ax, (col, color) in zip(axes, pairs):
        sns.histplot(df[col], ax=ax, bins=8, color=color, alpha=0.7,
                     edgecolor=DARK_BG, linewidth=0.5)
        ax.axvline(df[col].mean(), color=GREEN, lw=1.8, linestyle="--",
                   label=f"Mean {df[col].mean():.1f}")
        fig_style(ax, f"{col} Distribution", "Score", "# Students")
        ax.tick_params(axis="x", colors=MUTED)
        ax.legend(facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    fig.tight_layout(pad=2)
    return fig


def chart_attendance_pie(df: pd.DataFrame):
    att = df["Attendence"]
    labels = ["< 60%", "60–75%", "75–90%", "> 90%"]
    vals = [(att < 60).sum(),
        ((att >= 60) & (att < 75)).sum(),
        ((att >= 75) & (att < 90)).sum(),
        (att >= 90).sum(),]
    colors = [RED, AMBER, ACCENT, GREEN]
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(width=0.6, edgecolor=DARK_BG, linewidth=2))
    
    for t in texts:      
        t.set_color(TEXT)  
        t.set_fontsize(9)
    for t in autotexts:  
        t.set_color(DARK_BG)
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax.set_title("Attendance Distribution", color=TEXT, fontsize=11, fontweight="bold")
    return fig


def chart_scatter_pred(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=DARK_BG)
    ax.scatter(df["finals"], df["predicted_finals"],
               color=ACCENT, alpha=0.65, edgecolors=DARK_BG, s=55)
    lims = [min(df["finals"].min(), df["predicted_finals"].min()) - 5,
            max(df["finals"].max(), df["predicted_finals"].max()) + 5]
    ax.plot(lims, lims, color=AMBER, lw=1.5, linestyle="--", label="Perfect fit")
    fig_style(ax, "Actual vs Predicted Finals", "Actual", "Predicted")
    ax.legend(facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    fig.tight_layout(pad=1.5)
    return fig


def chart_student_vs_class(df: pd.DataFrame, reg_no):
    row = df[df["Register No"] == int(reg_no)]
    if row.empty:
        return None, None
    s= row.iloc[0]
    avgs= [df[c].mean() for c in EXAM_COLS]
    scores= [s[c] for c in EXAM_COLS]
    x= np.arange(len(EXAM_COLS))
    w= 0.32

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    ax.bar(x - w/2, scores, width=w, color=ACCENT,  alpha=0.85, label=s["Name"], edgecolor=ACCENT, linewidth=1)
    ax.bar(x + w/2, avgs,   width=w, color=PURPLE, alpha=0.65, label="Class Avg", edgecolor=PURPLE, linewidth=1)
    
    for bar, val in zip(ax.patches[:len(EXAM_COLS)], scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(int(val)), ha="center", color=TEXT, fontsize=9, fontweight="bold")
    for bar, val in zip(ax.patches[len(EXAM_COLS):], avgs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}", ha="center", color=TEXT, fontsize=9)
    fig_style(ax, f"{s['Name']}  vs  Class Average", ylabel="Marks")
    ax.set_xticks(x)
    ax.set_xticklabels(["CIA-1\n(/20)", "Mid-Sem\n(/50)", "CIA-3\n(/20)"], color=TEXT, fontsize=9)
    ax.legend(facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    fig.tight_layout(pad=1.5)
    return fig, s


def chart_coef(metrics: dict):
    coefs= metrics["coefs"]
    labels= list(coefs.keys())
    vals= list(coefs.values())
    colors= [GREEN if v >= 0 else RED for v in vals]

    fig, ax = plt.subplots(figsize=(7, 3), facecolor=DARK_BG)
    bars = ax.barh(labels, vals, color=colors, alpha=0.8, edgecolor=colors, linewidth=1)
    for bar, val in zip(bars, vals):
        ax.text(val + (0.2 if val >= 0 else -0.2),
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left" if val >= 0 else "right",
                color=TEXT, fontsize=9, fontweight="bold")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    fig_style(ax, "Model Coefficients (feature weights)", xlabel="Coefficient")
    ax.set_yticklabels(labels, color=TEXT, fontsize=9)
    fig.tight_layout(pad=1.5)
    return fig


def chart_attendance_vs_pred(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    ax.scatter(df["Attendence"], df["predicted_finals"],
               color=GREEN, alpha=0.65, edgecolors=DARK_BG, s=55)
    # trend line
    z= np.polyfit(df["Attendence"].dropna(), df["predicted_finals"][df["Attendence"].notna()], 1)
    p= np.poly1d(z)
    xs= np.linspace(df["Attendence"].min(), df["Attendence"].max(), 100)
    ax.plot(xs, p(xs), color=AMBER, lw=1.5, linestyle="--")
    fig_style(ax, "Attendance vs Predicted Finals", "Attendance %", "Predicted Final Marks")
    fig.tight_layout(pad=1.5)
    return fig


#====================================SIDERBAR & LAYOUT====================================
with st.sidebar:
    st.markdown("### 📁 Upload Files")
    train_file = st.file_uploader("Training Data CSV",type=["csv"],
        help="CSV with columns: Attendence, CIA-1, Mid-sem, CIA-3, finals",key="train_upload")
    
    analysis_file = st.file_uploader("Student Marks CSV (to analyse & predict)",
        type=["csv"], help="CSV with columns: Attendence, CIA-1, Mid-sem, CIA-3",
        key="analysis_upload")

    st.markdown("#### ℹ️ Mark Scales")
    st.markdown("""
| Exam    | Max  |
|---------|------|
| CIA-1   | 20   |
| Mid-Sem | 50   |
| CIA-3   | 20   |
| Finals  | 100  |
    """)
    st.markdown("---")


#Load default files if no upload
DEFAULT_TRAIN = "training_data.csv"
DEFAULT_ANALYSIS = "students marks data.csv"

def get_bytes(uploaded, default_path):
    if uploaded is not None:
        return uploaded.read()
    if os.path.exists(default_path):
        with open(default_path, "rb") as f:
            return f.read()
    return None

train_bytes = get_bytes(train_file,    DEFAULT_TRAIN)
analysis_bytes = get_bytes(analysis_file, DEFAULT_ANALYSIS)

if train_bytes is None:
    st.warning("⚠️  Please upload a **Training Data CSV** using the sidebar.")
    st.stop()
if analysis_bytes is None:
    st.warning("⚠️  Please upload a **Student Marks CSV** using the sidebar.")
    st.stop()

#Train model
model, metrics, df_train_scaled = train_model(train_bytes)

# Load analysis data & predict
df = load_csv(io.BytesIO(analysis_bytes))
df = scale_features(df)
X_pred = df[["att_sc", "cia1_sc", "mid_sc", "cia3_sc"]]
df["predicted_finals"] = model.predict(X_pred).clip(0, 100).round(1)


#==============================MAIN PAGE================================
st.markdown("# 🎓 Student Marks Analysis & Prediction")
st.markdown("---")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("👥 Students", len(df))
k2.metric("CIA-1 Avg", f"{df['CIA-1'].mean():.1f} / 20")
k3.metric("Mid-Sem Avg", f"{df['Mid-sem'].mean():.1f} / 50")
k4.metric("CIA-3 Avg", f"{df['CIA-3'].mean():.1f} / 20")
k5.metric("Predicted Avg", f"{df['predicted_finals'].mean():.1f} / 100")

st.markdown("---")
tab_overview, tab_charts, tab_predict, tab_student, tab_model = st.tabs([
    "📋 Overview", "📊 Charts", "🎯 Predictions", "🔍 Student Lookup", "🤖 Model Info"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("📄 All Students")
    display_df = df.copy()

    show_cols = ["Register No", "Name", "Attendence",
                 "CIA-1", "Mid-sem", "CIA-3", "predicted_finals"]
    existing  = [c for c in show_cols if c in display_df.columns]

    st.dataframe(display_df[existing].rename(columns={
            "Attendence": "Attendance %",
            "predicted_finals": "Predicted Finals"
        }).style.format({
            "Attendance %":"{:.1f}",
            "Predicted Finals":"{:.1f}",
            "CIA-1":"{:.1f}", "Mid-sem":"{:.1f}", "CIA-3":"{:.1f}",
            "Register No":"{:.0f}"
        }).background_gradient(
            subset=["Predicted Finals"],cmap="Blues"),
        use_container_width=True, hide_index=True,height=480)

    st.markdown(f"**{len(display_df)}** student(s) shown")

    st.markdown("---")
    st.subheader("📈 Descriptive Statistics")
    desc_cols = [c for c in ["CIA-1", "Mid-sem", "CIA-3"] if c in df.columns]
    st.dataframe(df[desc_cols].describe().round(2).loc[["count", "mean", "min", "max"]],
        use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · Charts
# ══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.subheader("📊 Class Average per Exam")
    st.pyplot(chart_class_averages(df), use_container_width=True)

    st.markdown("---")
    st.subheader("📉 Score Distributions")
    st.pyplot(chart_distributions(df), use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Attendance Categories")
        st.pyplot(chart_attendance_pie(df), use_container_width=True)
    with c2:
        st.subheader("📈 Attendance vs Predicted Finals")
        st.pyplot(chart_attendance_vs_pred(df), use_container_width=True)

    
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · Predictions
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("🎯 Predicted Final Marks")
    def grade(val):
        if val >= 90: return "O"
        elif val >= 80: return "A+"
        elif val >= 70: return "A"
        elif val >= 60: return "B+"
        elif val >= 50: return "B"
        elif val >= 40: return "C"
        else: return "F"

    pred_df = df.copy()
    pred_df["Grade"]  = pred_df["predicted_finals"].apply(grade)
    pred_df["Result"] = pred_df["predicted_finals"].apply(lambda v: "Pass ✅" if v >= 40 else "Fail ❌")

    show = ["Register No", "Name", "Attendence", "CIA-1", "Mid-sem", "CIA-3",
            "predicted_finals", "Grade", "Result"]
    show = [c for c in show if c in pred_df.columns]

    def color_pred(val):
        if val >= 75:  return "color: #10b981; font-weight:700"
        elif val >= 50: return "color: #f59e0b; font-weight:700"
        else: return "color: #ef4444; font-weight:700"

    styled = pred_df[show].rename(columns={
        "Attendence": "Attendance %",
        "predicted_finals": "Predicted Finals"
    }).style.applymap(color_pred, subset=["Predicted Finals"]).format({
        "Attendance %":"{:.1f}", "Predicted Finals": "{:.1f}",
        "CIA-1":"{:.1f}", "Mid-sem":"{:.1f}", "CIA-3":"{:.1f}",
        "Register No":"{:.0f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    st.markdown("---")
    st.subheader("📊 Grade Distribution")
    grade_counts = pred_df["Grade"].value_counts()
    grade_order  = ["O", "A+", "A", "B+", "B", "C", "F"]
    grade_colors = {"O": GREEN, "A+": ACCENT, "A": "#38bdf8",
                    "B+": PURPLE, "B": AMBER, "C": "#fb923c", "F": RED}

    fig, ax = plt.subplots(figsize=(8, 3), facecolor=DARK_BG)
    grades_present = [g for g in grade_order if g in grade_counts.index]
    vals   = [grade_counts.get(g, 0) for g in grades_present]
    colors = [grade_colors[g] for g in grades_present]
    bars   = ax.bar(grades_present, vals, color=colors, alpha=0.85, edgecolor=colors, linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha="center", color=TEXT, fontsize=10, fontweight="bold")
    fig_style(ax, "Predicted Grade Distribution", "Grade", "# Students")
    ax.set_xticklabels(grades_present, color=TEXT, fontsize=10)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)

    # Pass/Fail summary
    st.markdown("---")
    pass_count = (pred_df["predicted_finals"] >= 40).sum()
    fail_count = len(pred_df) - pass_count
    p1, p2, p3 = st.columns(3)
    p1.metric("✅ Pass",     pass_count)
    p2.metric("❌ Fail",     fail_count)
    p3.metric("📈 Pass Rate", f"{pass_count/len(pred_df)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · Student Lookup
# ══════════════════════════════════════════════════════════════════════════════
with tab_student:
    options = {
        f"{int(row['Register No'])} – {row['Name']}": int(row['Register No'])
        for _, row in df.iterrows()}
    
    selected_label = st.selectbox("Select a student", list(options.keys()))
    selected_reg   = options[selected_label]

    fig_cmp, student = chart_student_vs_class(df, selected_reg)
    if student is not None:
        st.markdown(f"### {student['Name']}")
        avgs = {c: df[c].mean() for c in EXAM_COLS}
        s1, s2, s3, s4, s5 = st.columns(5)

        def delta(val, avg):
            diff = val - avg
            return f"{'+' if diff >= 0 else ''}{diff:.1f}"

        s1.metric("CIA-1",    f"{student['CIA-1']} / 20",  delta(student['CIA-1'],   avgs['CIA-1']))
        s2.metric("Mid-Sem",  f"{student['Mid-sem']} / 50", delta(student['Mid-sem'], avgs['Mid-sem']))
        s3.metric("CIA-3",    f"{student['CIA-3']} / 20",  delta(student['CIA-3'],   avgs['CIA-3']))

        att_val = student["Attendence"]
        att_color = "normal" if att_val >= 75 else "inverse"
        s4.metric("Attendance", f"{att_val:.1f}%",
                  "✅ Sufficient" if att_val >= 75 else "⚠️ Below 75%",
                  delta_color=att_color)

        pred_val = student["predicted_finals"]
        s5.metric("Predicted Finals", f"{pred_val:.1f} / 100",grade(pred_val))

        st.markdown("---")
        st.pyplot(fig_cmp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · Model Info
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.subheader("🤖 Linear Regression Model")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",          metrics["MAE"],  help="Mean Absolute Error")
    m2.metric("R² Score",     metrics["R2"],   help="Coefficient of Determination")
    m3.metric("Train Samples",metrics["n_train"])
    m4.metric("Test Samples", metrics["n_test"])

    st.markdown("---")
    st.subheader("📐 Feature Coefficients")
    st.markdown("Each coefficient shows how much the predicted final mark changes per 1-unit increase in the scaled feature (0–100 range).")
    st.pyplot(chart_coef(metrics), use_container_width=True)

    coef_data = {
        "Feature": list(metrics["coefs"].keys()),
        "Coefficient": [round(v, 4) for v in metrics["coefs"].values()],}
    
    st.dataframe(pd.DataFrame(coef_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📊 Training Data: Actual vs Predicted")

    # Predict on training set for scatter
    df_tr = df_train_scaled.copy()
    df_tr["predicted_finals"] = model.predict(
        df_tr[["att_sc", "cia1_sc", "mid_sc", "cia3_sc"]]).clip(0, 100)
    st.pyplot(chart_scatter_pred(df_tr), use_container_width=True)

