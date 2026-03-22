import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="CKM 2D Phenotype Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# CSS styling
# =========================
CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 3.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.28;
    margin-top: 0.25rem;
    margin-bottom: 0.35rem;
    overflow: visible;
    word-break: break-word;
}

.sub-title {
    font-size: 1.0rem;
    color: #4b5563;
    line-height: 1.5;
    margin-bottom: 0.4rem;
}

.panel {
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px 20px;
    background: #ffffff;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
    margin-bottom: 14px;
}

.kpi-card {
    border-radius: 16px;
    padding: 14px 16px;
    background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 6px rgba(0,0,0,0.03);
}

.kpi-label {
    font-size: 0.90rem;
    color: #6b7280;
    margin-bottom: 0.25rem;
}

.kpi-value {
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 0.82rem;
    color: #6b7280;
    margin-top: 0.2rem;
}

.badge {
    display: inline-block;
    padding: 0.30rem 0.70rem;
    border-radius: 999px;
    font-size: 0.83rem;
    font-weight: 700;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    border: 1px solid transparent;
}

.badge-red {
    background: #fee2e2;
    color: #991b1b;
    border-color: #fecaca;
}

.badge-orange {
    background: #ffedd5;
    color: #9a3412;
    border-color: #fdba74;
}

.badge-blue {
    background: #dbeafe;
    color: #1d4ed8;
    border-color: #93c5fd;
}

.badge-green {
    background: #dcfce7;
    color: #166534;
    border-color: #86efac;
}

.badge-slate {
    background: #f3f4f6;
    color: #374151;
    border-color: #d1d5db;
}

.card {
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    margin-bottom: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}

.card-info {
    background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
    border-color: #bfdbfe;
}

.card-warning {
    background: linear-gradient(180deg, #fff7ed 0%, #ffffff 100%);
    border-color: #fed7aa;
}

.card-success {
    background: linear-gradient(180deg, #ecfdf5 0%, #ffffff 100%);
    border-color: #a7f3d0;
}

.card-danger {
    background: linear-gradient(180deg, #fef2f2 0%, #ffffff 100%);
    border-color: #fecaca;
}

.card-title {
    font-size: 1.02rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
}

.card-body {
    font-size: 0.93rem;
    line-height: 1.55;
    color: #1f2937;
}

.card-foot {
    font-size: 0.82rem;
    color: #6b7280;
    margin-top: 0.55rem;
}

.small-note {
    color: #6b7280;
    font-size: 0.85rem;
}

.section-title {
    font-size: 1.18rem;
    font-weight: 800;
    margin: 0.3rem 0 0.8rem 0;
}

.hr-soft {
    border: 0;
    height: 1px;
    background: #e5e7eb;
    margin: 1rem 0 1.2rem 0;
}

/* optional: slightly reduce top chrome interference */
div[data-testid="stToolbar"] {
    top: 0.4rem;
}

@media (max-width: 900px) {
    .block-container {
        padding-top: 2.2rem;
    }
    .main-title {
        font-size: 1.7rem;
        line-height: 1.32;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# Paths and data loading
# =========================
APP_DIR = Path(__file__).resolve().parent

CANDIDATE_DATA_DIRS = [
    APP_DIR / "app_data",
    APP_DIR / "output" / "app_export",
    Path("/mount/src/spise-and-epvs/app_data"),
    Path("/mount/src/spise-and-epvs/output/app_export"),
]

def resolve_data_dir():
    for d in CANDIDATE_DATA_DIRS:
        if d.exists():
            return d
    return CANDIDATE_DATA_DIRS[0]

DATA_DIR = resolve_data_dir()

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


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_all_data():
    missing = []
    out = {}

    for cohort, files in COHORT_FILES.items():
        for _, path in files.items():
            if not path.exists():
                missing.append(str(path))

    if missing:
        msg = "The following required files are missing:\n\n" + "\n".join(f"- {x}" for x in missing)
        raise FileNotFoundError(msg)

    for cohort, files in COHORT_FILES.items():
        out[cohort] = {
            "meta": load_json(files["meta"]),
            "points": load_csv(files["points"]),
            "risk": load_csv(files["risk"]),
        }
    return out


try:
    DATA = load_all_data()
except Exception as e:
    st.error("Data files could not be loaded.")
    st.code(str(e))
    st.stop()


# =========================
# Language support
# =========================
LANG_OPTIONS = {
    "Bilingual": "bi",
    "English": "en",
    "中文": "zh",
}

TEXT = {
    "title": {
        "en": "CKM 2D Phenotype Tool",
        "zh": "CKM 二维表型工具",
    },
    "subtitle": {
        "en": "Research-use phenotype mapping based on SPISE and ΔePVS, with guideline-aligned interpretation.",
        "zh": "基于 SPISE 和 ΔePVS 的研究型二维表型定位，并结合指南框架进行解释。",
    },
    "caption": {
        "en": "Supports clinician-patient discussion and does not replace diagnosis or prescribing decisions.",
        "zh": "用于辅助医患沟通，不替代临床诊断或处方决策。",
    },
    "reference_cohort": {
        "en": "Reference cohort",
        "zh": "参考队列",
    },
    "inputs": {
        "en": "Inputs",
        "zh": "输入参数",
    },
    "clinical_context": {
        "en": "Clinical context",
        "zh": "临床背景",
    },
    "classification": {
        "en": "Phenotype classification",
        "zh": "表型分型",
    },
    "risk_selected": {
        "en": "Risk in selected cohort",
        "zh": "所选参考队列中的风险信息",
    },
    "cross_cohort": {
        "en": "Cross-cohort comparison",
        "zh": "跨队列对比",
    },
    "guideline_title": {
        "en": "Guideline-aligned interpretation",
        "zh": "指南框架下的解释",
    },
    "safe_use": {
        "en": "How to use this tool safely",
        "zh": "如何安全使用本工具",
    },
}


def tr(key, mode="bi"):
    en = TEXT[key]["en"]
    zh = TEXT[key]["zh"]
    if mode == "en":
        return en
    if mode == "zh":
        return zh
    return f"{en} / {zh}"


def phenotype_label_map(mode="bi"):
    if mode == "en":
        return {
            "Type IV: Reference": "Type IV: Reference",
            "Type I: Metabolic-only": "Type I: Metabolic-only",
            "Type III: Non-metabolic congestion": "Type III: Non-metabolic congestion",
            "Type II: Malignant CKM": "Type II: Malignant CKM",
        }
    if mode == "zh":
        return {
            "Type IV: Reference": "IV型：参考型",
            "Type I: Metabolic-only": "I型：代谢主导型",
            "Type III: Non-metabolic congestion": "III型：非代谢性淤血型",
            "Type II: Malignant CKM": "II型：恶性 CKM 型",
        }
    return {
        "Type IV: Reference": "Type IV: Reference / IV型：参考型",
        "Type I: Metabolic-only": "Type I: Metabolic-only / I型：代谢主导型",
        "Type III: Non-metabolic congestion": "Type III: Non-metabolic congestion / III型：非代谢性淤血型",
        "Type II: Malignant CKM": "Type II: Malignant CKM / II型：恶性 CKM 型",
    }


def phenotype_short_label(phenotype, mode="bi"):
    return phenotype_label_map(mode).get(phenotype, phenotype)


# =========================
# Utility functions
# =========================
def tg_to_mgdl(value, unit):
    value = float(value)
    return value * 88.57 if unit == "mmol/L" else value


def hdl_to_mgdl(value, unit):
    value = float(value)
    return value * 38.67 if unit == "mmol/L" else value


def hb_to_gdl(value, unit):
    value = float(value)
    return value / 10.0 if unit == "g/L" else value


def hct_to_fraction(value, unit):
    value = float(value)
    return value / 100.0 if unit == "%" else value


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
    return "Type IV: Reference"


def percentile_rank(series, value):
    arr = pd.Series(series).dropna().to_numpy()
    if len(arr) == 0 or pd.isna(value):
        return np.nan
    return 100.0 * np.mean(arr <= value)


def phenotype_risk_row(risk_df, phenotype):
    x = risk_df.loc[risk_df["phenotype"] == phenotype]
    return None if x.empty else x


def format_hr(row):
    if row is None or row.empty:
        return "NA"
    hr = row["hr"].iloc[0]
    lo = row["hr_low"].iloc[0]
    hi = row["hr_high"].iloc[0]
    if pd.isna(hr):
        return "NA"
    return f"{hr:.2f} ({lo:.2f}-{hi:.2f})"


def bool_yes(x):
    return str(x).strip().lower() == "yes"


def egfr_category(egfr):
    if egfr >= 90:
        return "G1"
    elif egfr >= 60:
        return "G2"
    elif egfr >= 45:
        return "G3a"
    elif egfr >= 30:
        return "G3b"
    elif egfr >= 15:
        return "G4"
    return "G5"


def uacr_category(uacr):
    if uacr is None or uacr <= 0:
        return "Not available"
    if uacr < 30:
        return "A1"
    elif uacr <= 300:
        return "A2"
    return "A3"


def phenotype_badge_class(phenotype):
    mapping = {
        "Type II: Malignant CKM": "badge-red",
        "Type I: Metabolic-only": "badge-orange",
        "Type III: Non-metabolic congestion": "badge-blue",
        "Type IV: Reference": "badge-green",
    }
    return mapping.get(phenotype, "badge-slate")


def card_kind_for_phenotype(phenotype):
    mapping = {
        "Type II: Malignant CKM": "danger",
        "Type I: Metabolic-only": "warning",
        "Type III: Non-metabolic congestion": "info",
        "Type IV: Reference": "success",
    }
    return mapping.get(phenotype, "info")


def safe_pct(x):
    return "NA" if pd.isna(x) else f"{x:.1f}"


# =========================
# Narrative text
# =========================
PHENOTYPE_NOTES = {
    "Type IV: Reference": {
        "en": "Reference pattern: relatively preserved insulin sensitivity and no high ΔePVS signal.",
        "zh": "参考型：胰岛素敏感性相对保留，且未见较高 ΔePVS 信号。",
    },
    "Type I: Metabolic-only": {
        "en": "Metabolic vulnerability without a high congestion signal.",
        "zh": "以代谢脆弱性为主，但尚未表现出较高的容量/淤血信号。",
    },
    "Type III: Non-metabolic congestion": {
        "en": "Congestion-predominant pattern; risk may be weaker than Type II.",
        "zh": "以淤血/容量负荷信号为主；其风险通常弱于 II 型。",
    },
    "Type II: Malignant CKM": {
        "en": "Combined metabolic dysfunction and congestion signal; highest-risk phenotype.",
        "zh": "代谢异常与淤血信号并存，是风险最高的表型。",
    },
}

SAFE_USE_TEXT = {
    "en": """
1. The 2D phenotype is a research-derived risk-context tool, not a stand-alone diagnostic system.  
2. Guideline notes are intentionally conservative and should be read alongside standard care pathways.  
3. Kidney interpretation is stronger when both eGFR and urine ACR are available.  
4. ΔePVS requires two time points; a single visit cannot fully reproduce the phenotype framework.
""",
    "zh": """
1. 本工具属于研究型风险分层工具，不能替代独立诊断系统。  
2. 指南解释模块采用保守表述，应结合标准诊疗路径理解。  
3. 肾脏风险判断在同时具备 eGFR 和尿白蛋白/肌酐比时更可靠。  
4. ΔePVS 需要两个时间点，单次就诊不能完整重建该表型框架。
""",
}


# =========================
# Guideline cards
# =========================
def build_guideline_cards(
    phenotype,
    age,
    sex,
    smoker,
    sbp,
    diabetes,
    ascvd,
    egfr,
    uacr,
    lang_mode="bi",
):
    diabetes = bool_yes(diabetes)
    ascvd = bool_yes(ascvd)
    smoker = bool_yes(smoker)

    has_ckd_signal = (egfr < 60) or (uacr >= 30 and uacr > 0)
    egfr_cat = egfr_category(egfr)
    uacr_cat = uacr_category(uacr)

    cards = []

    if phenotype == "Type II: Malignant CKM":
        en = (
            "This patient falls in the malignant CKM phenotype zone (low SPISE / high ΔePVS). "
            "In the research cohorts, this was the highest-risk pattern. "
            "Treat this as a signal for closer cardio-renal-metabolic surveillance rather than as a stand-alone treatment trigger."
        )
        zh = (
            "该患者落在恶性 CKM 表型区域（低 SPISE / 高 ΔePVS）。在研究队列中，这一模式对应最高风险。"
            "应将其理解为需要更密切心-肾-代谢随访的信号，而不是单独的治疗触发条件。"
        )
        cards.append({
            "title_en": "2D phenotype interpretation",
            "title_zh": "二维表型解释",
            "kind": "danger",
            "body_en": en,
            "body_zh": zh,
        })
    elif phenotype == "Type III: Non-metabolic congestion":
        en = (
            "This patient falls in the non-metabolic congestion zone. "
            "Compared with the malignant phenotype, this usually suggests weaker risk concentration, "
            "but repeat assessment and volume-related follow-up remain important."
        )
        zh = (
            "该患者落在非代谢性淤血区域。与恶性 CKM 表型相比，这通常提示风险集中度较弱，"
            "但仍应重视复测和容量相关随访。"
        )
        cards.append({
            "title_en": "2D phenotype interpretation",
            "title_zh": "二维表型解释",
            "kind": "info",
            "body_en": en,
            "body_zh": zh,
        })
    elif phenotype == "Type I: Metabolic-only":
        en = (
            "This patient falls in the metabolic-only zone. "
            "Metabolic risk appears to dominate without a high ΔePVS signal. "
            "Standard prevention and metabolic risk-factor control remain central."
        )
        zh = (
            "该患者落在代谢主导区域。当前以代谢风险为主，尚未见明显高 ΔePVS 信号。"
            "标准预防和代谢危险因素控制仍是管理核心。"
        )
        cards.append({
            "title_en": "2D phenotype interpretation",
            "title_zh": "二维表型解释",
            "kind": "warning",
            "body_en": en,
            "body_zh": zh,
        })
    else:
        en = (
            "This patient falls in the reference zone of the 2D phenotype map. "
            "That does not exclude future risk; it means no high-risk discordance pattern is identified at this time."
        )
        zh = (
            "该患者落在二维表型图的参考区域。"
            "这并不意味着未来没有风险，只表示当前未识别出高风险失衡模式。"
        )
        cards.append({
            "title_en": "2D phenotype interpretation",
            "title_zh": "二维表型解释",
            "kind": "success",
            "body_en": en,
            "body_zh": zh,
        })

    if ascvd:
        en = (
            "Known ASCVD/HF/stroke/PAD is entered. This is a secondary-prevention context. "
            "Use the 2D phenotype as an adjunct for surveillance and communication, not as a replacement for standard secondary-prevention care."
        )
        zh = (
            "已录入 ASCVD/HF/卒中/PAD 病史，属于二级预防语境。"
            "二维表型可作为随访和沟通的补充工具，但不能替代标准二级预防管理。"
        )
    else:
        en = (
            "No established CVD is entered. This is mainly a primary-prevention context. "
            "Use a standard global cardiovascular risk tool in parallel and interpret the 2D phenotype as complementary context."
        )
        zh = (
            "未录入明确心血管疾病，主要属于一级预防语境。"
            "建议同时结合常规总体心血管风险评估工具，将二维表型作为补充信息理解。"
        )

    cards.append({
        "title_en": "AHA CKM / global risk note",
        "title_zh": "AHA CKM / 总体风险提示",
        "kind": "info",
        "body_en": en,
        "body_zh": zh,
    })

    if has_ckd_signal:
        en = (
            f"Kidney risk signal is present from entered data (eGFR {egfr:.0f}, albuminuria category {uacr_cat}). "
            f"Current kidney category is {egfr_cat}/{uacr_cat}. Confirm persistence over at least 3 months before labeling chronic CKD. "
            "Medication review, kidney monitoring, and albuminuria follow-up are reasonable next steps."
        )
        zh = (
            f"根据录入数据，存在肾脏风险信号（eGFR {egfr:.0f}，白蛋白尿分层 {uacr_cat}）。"
            f"当前肾脏分层为 {egfr_cat}/{uacr_cat}。在标注为慢性 CKD 前，建议确认至少持续 3 个月。"
            "下一步可考虑药物回顾、肾功能监测和白蛋白尿复查。"
        )
        kind = "warning"
    else:
        if uacr <= 0:
            en = (
                f"eGFR is {egfr:.0f} ({egfr_cat}), but urine ACR is not entered. "
                "Kidney risk is incompletely characterized without albuminuria data; consider obtaining urine ACR."
            )
            zh = (
                f"eGFR 为 {egfr:.0f}（{egfr_cat}），但未录入尿 ACR。"
                "缺少白蛋白尿信息时，肾脏风险评估并不完整，建议补测尿 ACR。"
            )
            kind = "info"
        else:
            en = (
                f"Entered kidney measures do not suggest clear CKD by eGFR/albuminuria alone "
                f"(eGFR {egfr:.0f}, {egfr_cat}; albuminuria {uacr_cat}). "
                "Continue routine kidney surveillance according to overall cardiometabolic risk."
            )
            zh = (
                f"根据当前录入的 eGFR 和白蛋白尿水平，尚不提示明确 CKD "
                f"（eGFR {egfr:.0f}, {egfr_cat}; 白蛋白尿 {uacr_cat}）。"
                "仍应根据整体心肾代谢风险进行常规肾脏随访。"
            )
            kind = "success"

    cards.append({
        "title_en": "KDIGO 2024 kidney note",
        "title_zh": "KDIGO 2024 肾脏提示",
        "kind": kind,
        "body_en": en,
        "body_zh": zh,
    })

    if diabetes:
        if has_ckd_signal or ascvd:
            en = (
                "Diabetes is entered together with CKD signal and/or established CVD. "
                "Review lipid management, blood pressure control, kidney surveillance, "
                "and eligibility for therapies with cardiorenal benefit under standard diabetes care pathways."
            )
            zh = (
                "已录入糖尿病，且同时存在 CKD 信号和/或既往 CVD。"
                "建议在标准糖尿病管理路径下，系统评估血脂管理、血压控制、肾脏监测，"
                "以及具有心肾获益治疗策略的适用性。"
            )
            kind = "warning"
        else:
            en = (
                "Diabetes is entered. Reinforce integrated cardiovascular risk management, "
                "kidney surveillance, and glucose-lowering strategy review according to comorbidity profile."
            )
            zh = (
                "已录入糖尿病。建议强化心血管综合风险管理、肾脏监测，"
                "并结合合并症情况评估降糖策略。"
            )
            kind = "info"

        cards.append({
            "title_en": "ADA / ESC diabetes-cardiovascular note",
            "title_zh": "ADA / ESC 糖尿病-心血管提示",
            "kind": kind,
            "body_en": en,
            "body_zh": zh,
        })

    lifestyle_parts_en = []
    lifestyle_parts_zh = []

    if smoker:
        lifestyle_parts_en.append("Current smoking is entered; smoking cessation support should be prioritized.")
        lifestyle_parts_zh.append("已录入当前吸烟，戒烟支持应优先纳入管理。")

    if sbp >= 140:
        lifestyle_parts_en.append(
            f"Systolic BP is {sbp:.0f} mmHg, which is clearly elevated; confirm with standardized measurements and review antihypertensive intensification."
        )
        lifestyle_parts_zh.append(
            f"收缩压 {sbp:.0f} mmHg，已明显升高；建议标准化复测，并评估是否需要强化降压治疗。"
        )
    elif sbp >= 130 and (diabetes or has_ckd_signal or ascvd):
        lifestyle_parts_en.append(
            f"Systolic BP is {sbp:.0f} mmHg in a higher-risk context; repeated measurement and closer BP optimization are reasonable."
        )
        lifestyle_parts_zh.append(
            f"在较高风险背景下，收缩压 {sbp:.0f} mmHg；建议复测并更密切优化血压管理。"
        )

    if not lifestyle_parts_en:
        lifestyle_parts_en.append(
            "Continue lifestyle-focused prevention: physical activity, weight management, healthy diet, and routine risk-factor follow-up."
        )
        lifestyle_parts_zh.append(
            "继续坚持生活方式干预：增加体力活动、体重管理、健康饮食以及常规危险因素随访。"
        )

    cards.append({
        "title_en": "Lifestyle / BP note",
        "title_zh": "生活方式 / 血压提示",
        "kind": "info",
        "body_en": " ".join(lifestyle_parts_en),
        "body_zh": " ".join(lifestyle_parts_zh),
    })

    next_steps_en = []
    next_steps_zh = []

    if uacr <= 0:
        next_steps_en.append("Obtain urine ACR")
        next_steps_zh.append("补测尿 ACR")
    if phenotype in ["Type II: Malignant CKM", "Type III: Non-metabolic congestion"]:
        next_steps_en.append("Repeat Hb/Hct and reassess ΔePVS over time")
        next_steps_zh.append("复测 Hb/Hct 并随时间重新评估 ΔePVS")
    if not ascvd and 30 <= age <= 79:
        next_steps_en.append("Pair phenotype with a standard primary-prevention risk estimate")
        next_steps_zh.append("结合常规一级预防风险评估工具")
    if sbp >= 130:
        next_steps_en.append("Recheck office/home BP")
        next_steps_zh.append("复测门诊或家庭血压")
    if diabetes:
        next_steps_en.append("Review diabetes complication surveillance")
        next_steps_zh.append("回顾糖尿病并发症监测")

    if next_steps_en:
        cards.append({
            "title_en": "Suggested next checks",
            "title_zh": "建议补充检查 / 下一步",
            "kind": "info",
            "body_en": " • ".join(next_steps_en),
            "body_zh": " • ".join(next_steps_zh),
        })

    return cards


# =========================
# HTML render helpers
# =========================
def format_multilang(en_text, zh_text, lang_mode="bi"):
    if lang_mode == "en":
        return en_text
    if lang_mode == "zh":
        return zh_text
    return f"{en_text}<br><span class='small-note'>{zh_text}</span>"


def render_badges_html(items):
    return "".join(items)


def render_html_card(title_en, title_zh, body_en, body_zh, kind="info", foot_en=None, foot_zh=None, lang_mode="bi"):
    title = title_en if lang_mode == "en" else title_zh if lang_mode == "zh" else f"{title_en} / {title_zh}"
    body = format_multilang(body_en, body_zh, lang_mode)
    foot_html = ""
    if foot_en or foot_zh:
        if lang_mode == "en":
            foot_html = f"<div class='card-foot'>{foot_en or ''}</div>"
        elif lang_mode == "zh":
            foot_html = f"<div class='card-foot'>{foot_zh or ''}</div>"
        else:
            foot_html = f"<div class='card-foot'>{foot_en or ''}<br>{foot_zh or ''}</div>"

    st.markdown(
        f"""
        <div class="card card-{kind}">
            <div class="card-title">{title}</div>
            <div class="card-body">{body}</div>
            {foot_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_html(label_en, label_zh, value, sub_en="", sub_zh="", lang_mode="bi"):
    label = label_en if lang_mode == "en" else label_zh if lang_mode == "zh" else f"{label_en} / {label_zh}"
    if lang_mode == "en":
        sub = sub_en
    elif lang_mode == "zh":
        sub = sub_zh
    else:
        sub = f"{sub_en}<br>{sub_zh}" if sub_en or sub_zh else ""

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Plot
# =========================
def build_plot(meta, points, patient_spise, patient_depvs, phenotype, lang_mode="bi"):
    x_cut = meta["x_cut"]
    y_cut = meta["y_cut"]
    x_min, x_max = meta["x_range"]
    y_min, y_max = meta["y_range"]
    colors = meta["phenotype_colors"]

    fig = go.Figure()

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
            opacity=0.14,
            layer="below",
        )

    fig.add_trace(
        go.Scattergl(
            x=points["spise_w1"],
            y=points["delta_epvs"],
            mode="markers",
            name="Reference population",
            marker=dict(size=4, color="rgba(80,80,80,0.16)"),
            hoverinfo="skip",
        )
    )

    fig.add_vline(x=x_cut, line_width=2, line_dash="dash", line_color="#374151")
    fig.add_hline(y=y_cut, line_width=2, line_dash="dash", line_color="#374151")

    fig.add_trace(
        go.Scatter(
            x=[patient_spise],
            y=[patient_depvs],
            mode="markers+text",
            name="Patient",
            text=["Patient"],
            textposition="top center",
            marker=dict(size=18, color="#dc2626", symbol="star"),
            hovertemplate="SPISE=%{x:.2f}<br>ΔePVS=%{y:.2f}<extra></extra>",
        )
    )

    labels = phenotype_label_map(lang_mode)

    fig.add_annotation(x=(x_min + x_cut) / 2, y=(y_min + y_cut) / 2, text=labels["Type I: Metabolic-only"].replace(" / ", "<br>"), showarrow=False)
    fig.add_annotation(x=(x_min + x_cut) / 2, y=(y_cut + y_max) / 2, text=labels["Type II: Malignant CKM"].replace(" / ", "<br>"), showarrow=False)
    fig.add_annotation(x=(x_cut + x_max) / 2, y=(y_cut + y_max) / 2, text=labels["Type III: Non-metabolic congestion"].replace(" / ", "<br>"), showarrow=False)
    fig.add_annotation(x=(x_cut + x_max) / 2, y=(y_min + y_cut) / 2, text=labels["Type IV: Reference"].replace(" / ", "<br>"), showarrow=False)

    title = meta["cohort_label"] if lang_mode == "en" else (
        "参考人群：" + meta["cohort_label"] if lang_mode == "zh" else f"2D phenotype map / 二维表型图 — {meta['cohort_label']}"
    )

    x_title = meta["x_label"] if lang_mode != "zh" else "SPISE（基线）"
    y_title = meta["y_label"] if lang_mode != "zh" else ("ΔePVS（W3 - W1）" if "W3" in meta["y_label"] else "ΔePVS（W2 - W1）")

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=680,
        legend=dict(orientation="h"),
        template="simple_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    return fig


# =========================
# Sidebar
# =========================
with st.sidebar:
    language_display = st.selectbox("Language / 语言", list(LANG_OPTIONS.keys()), index=0)
    lang_mode = LANG_OPTIONS[language_display]

    st.markdown("### " + tr("reference_cohort", lang_mode))
    selected_cohort = st.radio("Cohort", ["CHARLS", "UK Biobank"], index=0)

    st.markdown("### " + tr("inputs", lang_mode))
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=80.0, value=27.5, step=0.1)

    tg_unit = st.selectbox("Triglycerides unit", ["mmol/L", "mg/dL"], index=0)
    tg_value = st.number_input("Triglycerides", min_value=0.01, value=1.90, step=0.01)

    hdl_unit = st.selectbox("HDL-C unit", ["mmol/L", "mg/dL"], index=0)
    hdl_value = st.number_input("HDL-C", min_value=0.01, value=1.30, step=0.01)

    st.markdown("#### Baseline / 基线")
    hb1_unit = st.selectbox("Baseline Hb unit", ["g/dL", "g/L"], index=0)
    hb1_value = st.number_input("Baseline Hb", min_value=0.1, value=13.5, step=0.1)

    hct1_unit = st.selectbox("Baseline Hct unit", ["fraction", "%"], index=1)
    hct1_value = st.number_input("Baseline Hct", min_value=0.01, value=40.0, step=0.1)

    st.markdown("#### Follow-up / 随访")
    hb2_unit = st.selectbox("Follow-up Hb unit", ["g/dL", "g/L"], index=0)
    hb2_value = st.number_input("Follow-up Hb", min_value=0.1, value=13.0, step=0.1)

    hct2_unit = st.selectbox("Follow-up Hct unit", ["fraction", "%"], index=1)
    hct2_value = st.number_input("Follow-up Hct", min_value=0.01, value=38.5, step=0.1)

    st.markdown("### " + tr("clinical_context", lang_mode))
    age_clin = st.number_input("Age / 年龄", min_value=18, max_value=100, value=58, step=1)
    sex_clin = st.selectbox("Sex / 性别", ["Male", "Female"], index=0)
    smoker_clin = st.selectbox("Current smoker / 当前吸烟", ["No", "Yes"], index=0)
    sbp_clin = st.number_input("Systolic BP / 收缩压 (mmHg)", min_value=70, max_value=250, value=145, step=1)

    diabetes_clin = st.selectbox("Known diabetes / 已知糖尿病", ["No", "Yes"], index=0)
    ascvd_clin = st.selectbox("Known ASCVD/HF/stroke/PAD / 已知 ASCVD/HF/卒中/PAD", ["No", "Yes"], index=0)

    egfr_clin = st.number_input("eGFR (mL/min/1.73m²)", min_value=1.0, max_value=200.0, value=90.0, step=1.0)
    uacr_clin = st.number_input("Urine ACR / 尿 ACR (mg/g; 0 if unavailable)", min_value=0.0, max_value=5000.0, value=0.0, step=1.0)


# =========================
# Main calculations
# =========================
meta = DATA[selected_cohort]["meta"]
points = DATA[selected_cohort]["points"]
risk = DATA[selected_cohort]["risk"]

spise = calc_spise(tg_value, tg_unit, hdl_value, hdl_unit, bmi)
epvs1 = calc_epvs(hb1_value, hb1_unit, hct1_value, hct1_unit)
epvs2 = calc_epvs(hb2_value, hb2_unit, hct2_value, hct2_unit)
delta_epvs = epvs2 - epvs1 if np.isfinite(epvs1) and np.isfinite(epvs2) else np.nan

phenotype = classify_phenotype(spise, delta_epvs, meta["x_cut"], meta["y_cut"])
spise_pct = percentile_rank(points["spise_w1"], spise)
depvs_pct = percentile_rank(points["delta_epvs"], delta_epvs)
risk_row = phenotype_risk_row(risk, phenotype)

phenotype_note_en = PHENOTYPE_NOTES.get(phenotype, {}).get("en", "")
phenotype_note_zh = PHENOTYPE_NOTES.get(phenotype, {}).get("zh", "")

event_rate_pct = None
n_grp = None
events_grp = None
hr_text = "NA"
median_fu = None

if risk_row is not None:
    event_rate_pct = float(risk_row["event_rate_pct"].iloc[0])
    n_grp = int(risk_row["n"].iloc[0])
    events_grp = int(risk_row["events"].iloc[0])
    hr_text = format_hr(risk_row)
    median_fu = float(risk_row["median_fu_years"].iloc[0])


# =========================
# Header
# =========================
st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
st.markdown(f"<div class='main-title'>{tr('title', lang_mode)}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-title'>{tr('subtitle', lang_mode)}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='small-note' style='margin-bottom: 0.8rem;'>{tr('caption', lang_mode)}</div>", unsafe_allow_html=True)

# =========================
# KPI row
# =========================
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_html("SPISE", "SPISE", f"{spise:.2f}", "Insulin sensitivity surrogate", "胰岛素敏感性替代指标", lang_mode)
with k2:
    render_kpi_html("Baseline ePVS", "基线 ePVS", f"{epvs1:.2f}", "Estimated plasma volume status", "估算血浆容量状态", lang_mode)
with k3:
    render_kpi_html("Follow-up ePVS", "随访 ePVS", f"{epvs2:.2f}", "Estimated plasma volume status", "估算血浆容量状态", lang_mode)
with k4:
    render_kpi_html("ΔePVS", "ΔePVS", f"{delta_epvs:.2f}", "Change over time", "随时间变化值", lang_mode)

st.markdown("<hr class='hr-soft'>", unsafe_allow_html=True)


# =========================
# Main layout: plot + summary
# =========================
left, right = st.columns([1.8, 1.1])

with left:
    fig = build_plot(meta, points, spise, delta_epvs, phenotype, lang_mode)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown(f"<div class='section-title'>{tr('classification', lang_mode)}</div>", unsafe_allow_html=True)

    badge_cls = phenotype_badge_class(phenotype)
    st.markdown(
        f"<span class='badge {badge_cls}'>{phenotype_short_label(phenotype, lang_mode)}</span>",
        unsafe_allow_html=True,
    )

    render_html_card(
        "Phenotype summary",
        "表型摘要",
        phenotype_note_en,
        phenotype_note_zh,
        kind=card_kind_for_phenotype(phenotype),
        foot_en=f"Reference cohort: {meta['cohort_label']}",
        foot_zh=f"参考队列：{meta['cohort_label']}",
        lang_mode=lang_mode,
    )

    x_cut = meta["x_cut"]
    y_cut = meta["y_cut"]
    render_html_card(
        "Position on map",
        "在图中的位置",
        (
            f"SPISE percentile: {safe_pct(spise_pct)}; ΔePVS percentile: {safe_pct(depvs_pct)}. "
            f"Low-SPISE threshold in {meta['cohort_label']}: {x_cut:.2f}; high-ΔePVS threshold: {y_cut:.2f}."
        ),
        (
            f"SPISE 位于参考人群第 {safe_pct(spise_pct)} 百分位；ΔePVS 位于第 {safe_pct(depvs_pct)} 百分位。"
            f"{meta['cohort_label']} 中低 SPISE 阈值为 {x_cut:.2f}，高 ΔePVS 阈值为 {y_cut:.2f}。"
        ),
        kind="info",
        lang_mode=lang_mode,
    )

    if risk_row is not None:
        render_html_card(
            "Risk in selected cohort",
            "所选队列中的风险信息",
            (
                f"Observed event rate: {event_rate_pct:.1f}% ({events_grp}/{n_grp}). "
                f"Adjusted HR vs reference: {hr_text}. Median follow-up: {median_fu:.2f} years."
            ),
            (
                f"观察到的事件率：{event_rate_pct:.1f}%（{events_grp}/{n_grp}）。"
                f"相对于参考型的调整后 HR：{hr_text}。中位随访：{median_fu:.2f} 年。"
            ),
            kind="warning" if phenotype == "Type II: Malignant CKM" else "info",
            lang_mode=lang_mode,
        )


# =========================
# Cross-cohort comparison
# =========================
st.markdown("<hr class='hr-soft'>", unsafe_allow_html=True)
st.markdown(f"<div class='section-title'>{tr('cross_cohort', lang_mode)}</div>", unsafe_allow_html=True)

comparison_rows = []
for cohort_name in ["CHARLS", "UK Biobank"]:
    risk_df = DATA[cohort_name]["risk"]
    row = phenotype_risk_row(risk_df, phenotype)
    if row is not None and not row.empty:
        comparison_rows.append({
            "Cohort": cohort_name,
            "Phenotype": phenotype_short_label(phenotype, lang_mode),
            "Observed event rate (%)": float(row["event_rate_pct"].iloc[0]),
            "Adjusted HR": format_hr(row),
            "Median follow-up (years)": round(float(row["median_fu_years"].iloc[0]), 2),
        })

if comparison_rows:
    comp_df = pd.DataFrame(comparison_rows)
    if lang_mode == "zh":
        comp_df = comp_df.rename(columns={
            "Cohort": "队列",
            "Phenotype": "表型",
            "Observed event rate (%)": "事件率（%）",
            "Adjusted HR": "调整后 HR",
            "Median follow-up (years)": "中位随访（年）",
        })
    elif lang_mode == "bi":
        comp_df = comp_df.rename(columns={
            "Cohort": "Cohort / 队列",
            "Phenotype": "Phenotype / 表型",
            "Observed event rate (%)": "Observed event rate (%) / 事件率（%）",
            "Adjusted HR": "Adjusted HR / 调整后 HR",
            "Median follow-up (years)": "Median follow-up (years) / 中位随访（年）",
        })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# =========================
# Guideline-aligned interpretation
# =========================
st.markdown("<hr class='hr-soft'>", unsafe_allow_html=True)
st.markdown(f"<div class='section-title'>{tr('guideline_title', lang_mode)}</div>", unsafe_allow_html=True)

cards = build_guideline_cards(
    phenotype=phenotype,
    age=age_clin,
    sex=sex_clin,
    smoker=smoker_clin,
    sbp=sbp_clin,
    diabetes=diabetes_clin,
    ascvd=ascvd_clin,
    egfr=egfr_clin,
    uacr=uacr_clin,
    lang_mode=lang_mode,
)

for i in range(0, len(cards), 2):
    cols = st.columns(2)
    card1 = cards[i]
    with cols[0]:
        render_html_card(
            card1["title_en"],
            card1["title_zh"],
            card1["body_en"],
            card1["body_zh"],
            kind=card1["kind"],
            lang_mode=lang_mode,
        )
    if i + 1 < len(cards):
        card2 = cards[i + 1]
        with cols[1]:
            render_html_card(
                card2["title_en"],
                card2["title_zh"],
                card2["body_en"],
                card2["body_zh"],
                kind=card2["kind"],
                lang_mode=lang_mode,
            )


# =========================
# Footer notes
# =========================
st.markdown("<hr class='hr-soft'>", unsafe_allow_html=True)

render_html_card(
    "Important note",
    "重要说明",
    (
        "This app reproduces the research phenotype based on SPISE and ΔePVS. "
        "Because ΔePVS requires two time points, a single visit is not sufficient for full classification."
    ),
    (
        "本工具复现的是基于 SPISE 和 ΔePVS 的研究表型。"
        "由于 ΔePVS 需要两个时间点，因此单次就诊无法完成完整分型。"
    ),
    kind="info",
    lang_mode=lang_mode,
)

with st.expander(tr("safe_use", lang_mode)):
    if lang_mode == "en":
        st.markdown(SAFE_USE_TEXT["en"])
    elif lang_mode == "zh":
        st.markdown(SAFE_USE_TEXT["zh"])
    else:
        st.markdown(SAFE_USE_TEXT["en"])
        st.markdown("---")
        st.markdown(SAFE_USE_TEXT["zh"])
