import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "app_data"

COHORT_FILES = {
    "CHARLS": {
        "meta": DATA_DIR / "charls_meta.json",
        "points": DATA_DIR / "charls_points.csv.gz",
        "risk": DATA_DIR / "charls_phenotype_risk.csv",
    },
    "UK Biobank": {
        "meta": DATA_DIR / "ukb_meta.json",
        "points": DATA_DIR / "ukb_points.csv.gz",
        "risk": DATA_DIR / "ukb_phenotype_risk.csv",
    },
}

PHENOTYPE_NOTES = {
    "Type IV: Reference": "Reference pattern: relatively preserved insulin sensitivity and no high ΔePVS signal.",
    "Type I: Metabolic-only": "Metabolic vulnerability without high congestion signal.",
    "Type III: Non-metabolic congestion": "Congestion-predominant pattern; risk may be weaker than Type II.",
    "Type II: Malignant CKM": "Combined metabolic dysfunction and congestion signal; highest-risk phenotype.",
}


@st.cache_data
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path)


@st.cache_data
def load_all_data():
    out = {}
    for cohort, files in COHORT_FILES.items():
        out[cohort] = {
            "meta": load_json(files["meta"]),
            "points": load_csv(files["points"]),
            "risk": load_csv(files["risk"]),
        }
    return out


def tg_to_mgdl(value, unit):
    value = float(value)
    if unit == "mmol/L":
        return value * 88.57
    return value


def hdl_to_mgdl(value, unit):
    value = float(value)
    if unit == "mmol/L":
        return value * 38.67
    return value


def hb_to_gdl(value, unit):
    value = float(value)
    if unit == "g/L":
        return value / 10.0
    return value


def hct_to_fraction(value, unit):
    value = float(value)
    if unit == "%":
        return value / 100.0
    return value


def calc_spise(tg_value, tg_unit, hdl_value, hdl_unit, bmi):
    tg_mgdl = tg_to_mgdl(tg_value, tg_unit)
    hdl_mgdl = hdl_to_mgdl(hdl_value, hdl_unit)
    bmi = float(bmi)
    if tg_mgdl <= 0 or hdl_mgdl <= 0 or bmi <= 0:
        return np.nan
    return 600 * (hdl_mgdl ** 0.185) / ((tg_mgdl ** 0.2) * (bmi ** 1.338))


def calc_epvs(hb_value, hb_unit, hct_value, hct_unit):
    hb_gdl = hb_to_gdl(hb_value, hb_unit)
    hct_frac = hct_to_fraction(hct_value, hct_unit)
    if hb_gdl <= 0:
        return np.nan
    return 100 * (1 - hct_frac) / hb_gdl


def classify_phenotype(spise, delta_epvs, x_cut, y_cut):
    if pd.isna(spise) or pd.isna(delta_epvs):
        return None
    if spise <= x_cut and delta_epvs < y_cut:
        return "Type I: Metabolic-only"
    elif spise <= x_cut and delta_epvs >= y_cut:
        return "Type II: Malignant CKM"
    elif spise > x_cut and delta_epvs >= y_cut:
        return "Type III: Non-metabolic congestion"
    else:
        return "Type IV: Reference"


def percentile_rank(series, value):
    arr = pd.Series(series).dropna().to_numpy()
    if len(arr) == 0 or pd.isna(value):
        return np.nan
    return 100.0 * np.mean(arr <= value)


def format_hr(row):
    if row is None or row.empty:
        return "NA"
    hr = row["hr"].iloc[0]
    lo = row["hr_low"].iloc[0]
    hi = row["hr_high"].iloc[0]
    if pd.isna(hr):
        return "NA"
    return f"{hr:.2f} ({lo:.2f}-{hi:.2f})"


def phenotype_risk_row(risk_df, phenotype):
    x = risk_df.loc[risk_df["phenotype"] == phenotype]
    if x.empty:
        return None
    return x


def build_plot(meta, points, patient_spise, patient_depvs, phenotype):
    x_cut = meta["x_cut"]
    y_cut = meta["y_cut"]
    x_min, x_max = meta["x_range"]
    y_min, y_max = meta["y_range"]

    colors = meta["phenotype_colors"]

    fig = go.Figure()

    # Quadrant background
    quadrants = [
        ("Type I: Metabolic-only", x_min, x_cut, y_min, y_cut),
        ("Type II: Malignant CKM", x_min, x_cut, y_cut, y_max),
        ("Type IV: Reference", x_cut, x_max, y_min, y_cut),
        ("Type III: Non-metabolic congestion", x_cut, x_max, y_cut, y_max),
    ]

    for label, x0, x1, y0, y1 in quadrants:
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(width=0),
            fillcolor=colors[label],
            opacity=0.12,
            layer="below",
        )

    # Reference cloud
    fig.add_trace(
        go.Scattergl(
            x=points["spise_w1"],
            y=points["delta_epvs"],
            mode="markers",
            name="Reference population",
            marker=dict(size=4, color="rgba(80,80,80,0.18)"),
            hoverinfo="skip",
        )
    )

    # Threshold lines
    fig.add_vline(x=x_cut, line_width=2, line_dash="dash", line_color="#444444")
    fig.add_hline(y=y_cut, line_width=2, line_dash="dash", line_color="#444444")

    # Patient point
    fig.add_trace(
        go.Scatter(
            x=[patient_spise],
            y=[patient_depvs],
            mode="markers+text",
            name="Patient",
            text=["Patient"],
            textposition="top center",
            marker=dict(size=16, color="#D62728", symbol="star"),
            hovertemplate="SPISE=%{x:.2f}<br>ΔePVS=%{y:.2f}<extra></extra>",
        )
    )

    # Quadrant labels
    fig.add_annotation(x=(x_min + x_cut) / 2, y=(y_min + y_cut) / 2, text="Type I<br>Metabolic-only", showarrow=False)
    fig.add_annotation(x=(x_min + x_cut) / 2, y=(y_cut + y_max) / 2, text="Type II<br>Malignant CKM", showarrow=False)
    fig.add_annotation(x=(x_cut + x_max) / 2, y=(y_cut + y_max) / 2, text="Type III<br>Non-metabolic congestion", showarrow=False)
    fig.add_annotation(x=(x_cut + x_max) / 2, y=(y_min + y_cut) / 2, text="Type IV<br>Reference", showarrow=False)

    fig.update_layout(
        title=f"2D phenotype map — {meta['cohort_label']}",
        xaxis_title=meta["x_label"],
        yaxis_title=meta["y_label"],
        height=650,
        legend=dict(orientation="h"),
        template="simple_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    return fig


st.set_page_config(page_title="CKM 2D Phenotype Tool", layout="wide")
st.title("CKM 2D Phenotype Tool")
st.caption("Research-use phenotype mapping based on SPISE and ΔePVS. Not for stand-alone clinical decisions.")

data = load_all_data()

with st.sidebar:
    st.header("Reference cohort")
    selected_cohort = st.radio("Choose cohort", ["CHARLS", "UK Biobank"], index=0)

    st.header("Lipid / anthropometric input")
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=80.0, value=27.5, step=0.1)

    tg_unit = st.selectbox("Triglycerides unit", ["mmol/L", "mg/dL"], index=0)
    tg_value = st.number_input("Triglycerides", min_value=0.01, value=1.90, step=0.01)

    hdl_unit = st.selectbox("HDL-C unit", ["mmol/L", "mg/dL"], index=0)
    hdl_value = st.number_input("HDL-C", min_value=0.01, value=1.30, step=0.01)

    st.header("Baseline blood counts")
    hb1_unit = st.selectbox("Baseline Hb unit", ["g/dL", "g/L"], index=0)
    hb1_value = st.number_input("Baseline Hb", min_value=0.1, value=13.5, step=0.1)

    hct1_unit = st.selectbox("Baseline Hct unit", ["fraction", "%"], index=1)
    hct1_value = st.number_input("Baseline Hct", min_value=0.01, value=40.0, step=0.1)

    st.header("Follow-up blood counts")
    hb2_unit = st.selectbox("Follow-up Hb unit", ["g/dL", "g/L"], index=0)
    hb2_value = st.number_input("Follow-up Hb", min_value=0.1, value=13.0, step=0.1)

    hct2_unit = st.selectbox("Follow-up Hct unit", ["fraction", "%"], index=1)
    hct2_value = st.number_input("Follow-up Hct", min_value=0.01, value=38.5, step=0.1)

meta = data[selected_cohort]["meta"]
points = data[selected_cohort]["points"]
risk = data[selected_cohort]["risk"]

spise = calc_spise(tg_value, tg_unit, hdl_value, hdl_unit, bmi)
epvs1 = calc_epvs(hb1_value, hb1_unit, hct1_value, hct1_unit)
epvs2 = calc_epvs(hb2_value, hb2_unit, hct2_value, hct2_unit)
delta_epvs = epvs2 - epvs1 if np.isfinite(epvs1) and np.isfinite(epvs2) else np.nan

phenotype = classify_phenotype(spise, delta_epvs, meta["x_cut"], meta["y_cut"])

spise_pct = percentile_rank(points["spise_w1"], spise)
depvs_pct = percentile_rank(points["delta_epvs"], delta_epvs)

col1, col2, col3, col4 = st.columns(4)
col1.metric("SPISE", f"{spise:.2f}")
col2.metric("Baseline ePVS", f"{epvs1:.2f}")
col3.metric("Follow-up ePVS", f"{epvs2:.2f}")
col4.metric("ΔePVS", f"{delta_epvs:.2f}")

st.divider()

left, right = st.columns([1.7, 1])

with left:
    fig = build_plot(meta, points, spise, delta_epvs, phenotype)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Phenotype classification")
    if phenotype is None:
        st.error("Unable to classify phenotype.")
    else:
        st.success(phenotype)
        st.write(PHENOTYPE_NOTES.get(phenotype, ""))

        st.write(
            f"""
            **Reference cohort:** {meta['cohort_label']}  
            **Low SPISE threshold:** {meta['x_cut']:.2f}  
            **High ΔePVS threshold:** {meta['y_cut']:.2f}  
            **SPISE percentile:** {spise_pct:.1f}  
            **ΔePVS percentile:** {depvs_pct:.1f}
            """
        )

        row = phenotype_risk_row(risk, phenotype)
        if row is not None:
            event_pct = row["event_rate_pct"].iloc[0]
            n = int(row["n"].iloc[0])
            events = int(row["events"].iloc[0])
            hr_text = format_hr(row)

            st.subheader("Risk in selected cohort")
            st.write(
                f"""
                **Observed event rate:** {event_pct:.1f}%  
                **Events / N:** {events} / {n}  
                **Adjusted HR vs reference:** {hr_text}
                """
            )

st.divider()

st.subheader("Cross-cohort comparison")
comparison_rows = []
for cohort_name in ["CHARLS", "UK Biobank"]:
    risk_df = data[cohort_name]["risk"]
    row = phenotype_risk_row(risk_df, phenotype)
    if row is not None and not row.empty:
        comparison_rows.append({
            "Cohort": cohort_name,
            "Phenotype": phenotype,
            "Observed event rate (%)": float(row["event_rate_pct"].iloc[0]),
            "Adjusted HR": format_hr(row),
            "Median follow-up (years)": round(float(row["median_fu_years"].iloc[0]), 2),
        })

if comparison_rows:
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

st.info(
    "Important: this app reproduces the research phenotype based on SPISE and ΔePVS. "
    "Because ΔePVS requires two time points, a single visit is not sufficient for full classification."
)