# -*- coding: utf-8 -*-
"""
Phân tích thái độ đối với công nghệ nhận diện khuôn mặt
=======================================================
Ứng dụng Streamlit (chuyển đổi từ bản Gradio chạy trong Google Colab).

Cách chạy:
    pip install -r requirements.txt
    streamlit run app.py

Dữ liệu: tải file Excel khảo sát (.xlsx) lên ở ô "Chọn file Excel khảo sát",
hoặc để trống để ứng dụng tự dò tìm file *.xlsx trong thư mục chạy ứng dụng.

Các biến nghiên cứu:  privacy_risk, support, scenario, manipulation_check
"""

import tempfile

import streamlit as st

st.set_page_config(
    page_title="Phân tích thái độ đối với công nghệ nhận diện khuôn mặt",
    page_icon="🎭",
    layout="wide",
)

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import os
import re
import glob
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.family"] = "DejaVu Sans"   # hỗ trợ tiếng Việt có dấu

SAMPLE_WARNING = (
    "⚠️ **Lưu ý: File dữ liệu hiện tại là dữ liệu mẫu dùng để kiểm tra ứng dụng, "
    "không phải kết quả khảo sát thực tế.**"
)

# Đường dẫn mặc định sẽ được dò tìm tự động
DEFAULT_FILE_PATTERNS = [
    "/content/*facial*recognition*.xlsx",
    "/content/drive/MyDrive/*facial*recognition*.xlsx",
    "./*facial*recognition*.xlsx",
    "/content/*.xlsx",
    "./*.xlsx",
]


# =============================================================================
# 2. LOAD EXCEL DATA
# =============================================================================
def normalize_text(s):
    """Chuẩn hoá chuỗi: bỏ dấu nháy cong, hạ chữ thường, gộp khoảng trắng."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("’", "'").replace("‘", "'")
           .replace("“", '"').replace("”", '"')
           .replace("–", "-").replace("—", "-")
           .replace(" ", " "))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def find_default_file():
    for pattern in DEFAULT_FILE_PATTERNS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_excel(file_path=None):
    """Đọc sheet phản hồi của file Excel khảo sát. Trả về (DataFrame, đường dẫn)."""
    if not file_path:
        file_path = find_default_file()
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(
            "Không tìm thấy file Excel. Hãy tải file lên Colab hoặc chọn file "
            "ở ô 'Chọn file Excel' trên giao diện."
        )

    xls = pd.ExcelFile(file_path, engine="openpyxl")
    # Chọn sheet phản hồi (bỏ qua sheet README / ghi chú)
    sheet = xls.sheet_names[0]
    for name in xls.sheet_names:
        if "readme" not in name.lower() and "note" not in name.lower():
            sheet = name
            break

    # mangle_dupe_cols: các cột support bị trùng tên sẽ thành ".1", ".2"
    df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
    df = df.dropna(how="all").reset_index(drop=True)
    return df, file_path


# -----------------------------------------------------------------------------
# Nhận diện cột theo nội dung câu hỏi (không phụ thuộc vị trí cột cứng)
# -----------------------------------------------------------------------------
PRIVACY_ITEM_KEYS = [
    "serious threat to people's privacy",
    "worried that my face could be scanned",
    "face data collected by the police will be misused",
]

SUPPORT_ITEM_KEYS = [
    "i would support the police using facial recognition",
    "the police should be allowed to use facial recognition",
    "this is an acceptable use of facial recognition",
]

CONSENT_KEY = "do you agree to take part"
ATTENTION_KEY = "please select 2"
MONTH_KEY = "which month were you born"
MANIP_KEY = "what were the police using facial recognition for"

SCENARIO_LABELS = {
    1: "Nhận diện người mất tích",
    2: "Giám sát chung / phòng ngừa tội phạm",
    3: "Giám sát biểu tình / hoạt động chính trị",
}
SCENARIO_LABELS_EN = {
    1: "Missing-person identification",
    2: "General surveillance / crime prevention",
    3: "Protest monitoring",
}

# Tháng sinh -> block scenario
MONTH_TO_SCENARIO = {
    "january": 1, "april": 1, "july": 1, "october": 1,
    "february": 2, "may": 2, "august": 2, "november": 2,
    "march": 3, "june": 3, "september": 3, "december": 3,
    # phòng trường hợp dữ liệu ghi tháng bằng tiếng Việt hoặc dạng số
    "tháng 1": 1, "tháng 4": 1, "tháng 7": 1, "tháng 10": 1,
    "tháng 2": 2, "tháng 5": 2, "tháng 8": 2, "tháng 11": 2,
    "tháng 3": 3, "tháng 6": 3, "tháng 9": 3, "tháng 12": 3,
}
MONTH_NUM_TO_SCENARIO = {1: 1, 4: 1, 7: 1, 10: 1,
                         2: 2, 5: 2, 8: 2, 11: 2,
                         3: 3, 6: 3, 9: 3, 12: 3}

# Đáp án đúng của manipulation check theo scenario
CORRECT_MANIP = {
    1: "finding missing people",
    2: "identifying people in a busy area to help prevent crime",
    3: "identifying people taking part in a protest",
}


def detect_columns(df):
    """Trả về dict mô tả vị trí các cột cần dùng."""
    norm = {col: normalize_text(col) for col in df.columns}

    def find_one(key):
        for col, n in norm.items():
            if key in n:
                return col
        return None

    def find_all(key):
        return [col for col, n in norm.items() if key in n]

    cols = {}
    cols["consent"] = find_one(CONSENT_KEY)
    cols["attention"] = find_one(ATTENTION_KEY)
    cols["month"] = find_one(MONTH_KEY)

    # Privacy risk items
    privacy_cols = []
    for key in PRIVACY_ITEM_KEYS:
        c = find_one(key)
        if c is not None:
            privacy_cols.append(c)
    cols["privacy_items"] = privacy_cols

    # Manipulation check: 3 cột (một cho mỗi block), theo thứ tự cột trong file
    manip_cols = find_all(MANIP_KEY)
    order = list(df.columns)
    manip_cols = sorted(manip_cols, key=lambda c: order.index(c))
    cols["manip_cols"] = manip_cols

    # Support items: mỗi câu hỏi xuất hiện 3 lần (3 block).
    # Gom theo thứ tự cột -> block 1, 2, 3.
    support_by_key = {}
    for key in SUPPORT_ITEM_KEYS:
        found = sorted(find_all(key), key=lambda c: order.index(c))
        support_by_key[key] = found

    support_blocks = {}
    for block in (1, 2, 3):
        block_cols = []
        for key in SUPPORT_ITEM_KEYS:
            found = support_by_key.get(key, [])
            if len(found) >= block:
                block_cols.append(found[block - 1])
        support_blocks[block] = block_cols
    cols["support_blocks"] = support_blocks

    # Nhân khẩu học (chỉ dùng cho mô tả, không đưa vào mô hình chính)
    cols["age"] = find_one("what is your age")
    cols["gender"] = find_one("what is your gender")
    cols["occupation"] = find_one("which best describes you")
    cols["familiarity"] = find_one("how familiar were you with facial recognition")

    missing = []
    if cols["consent"] is None:
        missing.append("cột consent")
    if cols["attention"] is None:
        missing.append("cột attention check")
    if cols["month"] is None:
        missing.append("cột tháng sinh")
    if len(cols["privacy_items"]) < 2:
        missing.append("các item Privacy Risk")
    if any(len(v) < 3 for v in support_blocks.values()):
        missing.append("đủ 3 block support items")
    if len(manip_cols) < 3:
        missing.append("3 cột manipulation check")
    cols["missing"] = missing
    return cols


# =============================================================================
# 3. DATA CLEANING  +  4. VARIABLE CONSTRUCTION
# =============================================================================
def to_likert(value):
    """Chuyển câu trả lời Likert về số 1-5 (chấp nhận cả dạng chữ)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if 1 <= v <= 5 else np.nan
    s = normalize_text(value)
    m = re.match(r"^([1-5])\b", s)
    if m:
        return float(m.group(1))
    text_map = {
        "strongly disagree": 1.0, "disagree": 2.0, "neutral": 3.0,
        "neither agree nor disagree": 3.0, "agree": 4.0, "strongly agree": 5.0,
        "rất không đồng ý": 1.0, "không đồng ý": 2.0, "trung lập": 3.0,
        "đồng ý": 4.0, "rất đồng ý": 5.0,
    }
    for k, v in text_map.items():
        if s == k:
            return v
    for k, v in text_map.items():
        if k in s:
            return v
    return np.nan


def month_to_scenario(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return MONTH_NUM_TO_SCENARIO.get(int(value), np.nan)
    s = normalize_text(value)
    for name, sc in MONTH_TO_SCENARIO.items():
        if s.startswith(name) or name in s:
            return sc
    m = re.search(r"(\d{1,2})", s)
    if m:
        return MONTH_NUM_TO_SCENARIO.get(int(m.group(1)), np.nan)
    return np.nan


def is_consent_yes(value):
    if pd.isna(value):
        return False
    s = normalize_text(value)
    if s.startswith("no") or "không đồng ý" in s:
        return False
    return s.startswith("yes") or "agree" in s or "đồng ý" in s


def clean_and_build(df_raw, cols):
    """Làm sạch dữ liệu và tạo các biến nghiên cứu. Trả về (clean_data, summary)."""
    n_initial = len(df_raw)
    df = df_raw.copy()

    # --- Consent -------------------------------------------------------------
    consent_ok = df[cols["consent"]].apply(is_consent_yes)
    n_dropped_consent = int((~consent_ok).sum())
    df = df[consent_ok].copy()

    # --- Attention check (phải chọn đúng 2) ----------------------------------
    attention_value = df[cols["attention"]].apply(to_likert)
    attention_ok = attention_value == 2
    n_dropped_attention = int((~attention_ok).sum())
    df = df[attention_ok].copy()

    # --- Scenario ------------------------------------------------------------
    df["scenario_code"] = df[cols["month"]].apply(month_to_scenario)
    n_dropped_scenario = int(df["scenario_code"].isna().sum())
    df = df[df["scenario_code"].notna()].copy()
    df["scenario_code"] = df["scenario_code"].astype(int)
    df["scenario"] = df["scenario_code"].map(SCENARIO_LABELS)
    df["scenario_en"] = df["scenario_code"].map(SCENARIO_LABELS_EN)

    # --- privacy_risk --------------------------------------------------------
    privacy_numeric = df[cols["privacy_items"]].apply(lambda c: c.map(to_likert))
    df["privacy_risk"] = privacy_numeric.mean(axis=1, skipna=True)

    # --- support (lấy đúng block theo scenario, KHÔNG trộn block) ------------
    support_values = []
    manip_answers = []
    for idx, row in df.iterrows():
        block = int(row["scenario_code"])
        block_cols = cols["support_blocks"][block]
        items = [to_likert(row[c]) for c in block_cols]
        items = [v for v in items if not pd.isna(v)]
        support_values.append(np.mean(items) if items else np.nan)
        manip_col = cols["manip_cols"][block - 1]
        manip_answers.append(row[manip_col])
    df["support"] = support_values
    df["manip_answer"] = manip_answers

    # --- manipulation_check (1 = đúng, 0 = sai) ------------------------------
    def check_manip(row):
        ans = normalize_text(row["manip_answer"])
        if ans == "":
            return 0
        return 1 if ans == CORRECT_MANIP[int(row["scenario_code"])] else 0

    df["manipulation_check"] = df.apply(check_manip, axis=1)

    # --- Bỏ các dòng thiếu biến phân tích chính ------------------------------
    before_missing = len(df)
    df = df[df["privacy_risk"].notna() & df["support"].notna()].copy()
    n_dropped_missing = before_missing - len(df)

    # --- Cronbach's alpha (độ tin cậy thang đo) ------------------------------
    alpha_privacy = cronbach_alpha(privacy_numeric.loc[df.index])

    clean_data = df.reset_index(drop=True)
    summary = {
        "n_initial": n_initial,
        "n_dropped_consent": n_dropped_consent,
        "n_dropped_attention": n_dropped_attention,
        "n_dropped_scenario": n_dropped_scenario,
        "n_dropped_missing": n_dropped_missing,
        "n_clean": len(clean_data),
        "manip_accuracy": float(clean_data["manipulation_check"].mean()) if len(clean_data) else np.nan,
        "n_manip_correct": int(clean_data["manipulation_check"].sum()) if len(clean_data) else 0,
        "alpha_privacy": alpha_privacy,
        "privacy_items": cols["privacy_items"],
    }
    return clean_data, summary


def cronbach_alpha(items_df):
    """Cronbach's alpha cho một thang đo (bỏ qua nếu < 2 item hoặc < 3 quan sát)."""
    d = items_df.dropna()
    k = d.shape[1]
    if k < 2 or len(d) < 3:
        return np.nan
    item_var = d.var(axis=0, ddof=1).sum()
    total_var = d.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_var / total_var)


# =============================================================================
# 5. CORRELATION ANALYSIS
# =============================================================================
def significance_label(p):
    if pd.isna(p):
        return "Không xác định"
    if p < 0.001:
        return "Có ý nghĩa thống kê ở mức p < 0.001 (***)"
    if p < 0.01:
        return "Có ý nghĩa thống kê ở mức p < 0.01 (**)"
    if p < 0.05:
        return "Có ý nghĩa thống kê ở mức p < 0.05 (*)"
    if p < 0.10:
        return "Gần ngưỡng ý nghĩa (p < 0.10)"
    return "Không có ý nghĩa thống kê ở mức p < 0.05"


def fmt_p(p):
    if pd.isna(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def run_correlation(clean_data):
    x = clean_data["privacy_risk"]
    y = clean_data["support"]
    r, p = stats.pearsonr(x, y)
    n = len(clean_data)
    direction = "Dương (Positive)" if r > 0 else ("Âm (Negative)" if r < 0 else "Không có hướng")
    return {"r": r, "p": p, "n": n, "direction": direction}


def correlation_markdown(res):
    r = res["r"]
    md = []
    md.append("## 🔗 Phân tích tương quan\n")
    md.append("**Ma trận tương quan (Pearson r)**\n")
    md.append("| Variable | Privacy Risk | Support |")
    md.append("| --- | ---: | ---: |")
    md.append(f"| Privacy Risk | 1.000 | {r:.3f} |")
    md.append(f"| Support | {r:.3f} | 1.000 |")
    md.append("")
    md.append("**Kết quả kiểm định tương quan Pearson**\n")
    md.append("| Chỉ số | Giá trị |")
    md.append("| --- | ---: |")
    md.append(f"| Hệ số tương quan r | **{r:.3f}** |")
    md.append(f"| p-value | **{fmt_p(res['p'])}** |")
    md.append(f"| Số quan sát N | {res['n']} |")
    md.append(f"| Hướng của mối quan hệ | {res['direction']} |")
    md.append("")
    md.append(f"**Mức ý nghĩa:** {significance_label(res['p'])}")
    md.append("")
    md.append(
        "> *Giả thuyết H1 dự đoán mối quan hệ **âm** giữa nhận thức rủi ro quyền riêng tư "
        "và mức độ ủng hộ. Kết quả thống kê được trình bày ở trên để người đọc tự đánh giá; "
        "ứng dụng không tự kết luận chấp nhận hay bác bỏ giả thuyết.*"
    )
    return "\n".join(md)


# =============================================================================
# 6. REGRESSION ANALYSIS
# =============================================================================
def run_regression(clean_data):
    """support = β0 + β1 * privacy_risk + ε  (OLS, statsmodels)."""
    X = sm.add_constant(clean_data["privacy_risk"])
    y = clean_data["support"]
    model = sm.OLS(y, X).fit()
    b0 = model.params["const"]
    b1 = model.params["privacy_risk"]
    sign = "-" if b1 < 0 else "+"
    equation = f"Support = {b0:.2f} {sign} {abs(b1):.2f} × Privacy Risk"
    return {
        "model": model,
        "b0": b0,
        "b1": b1,
        "se_const": model.bse["const"],
        "se_b1": model.bse["privacy_risk"],
        "t_const": model.tvalues["const"],
        "t_b1": model.tvalues["privacy_risk"],
        "p_const": model.pvalues["const"],
        "p_b1": model.pvalues["privacy_risk"],
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "n": int(model.nobs),
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "equation": equation,
    }


def regression_markdown(res):
    md = []
    md.append("## 📉 Hồi quy tuyến tính đơn\n")
    md.append("**Mô hình:** `support = β0 + β1 × privacy_risk + ε`\n")
    md.append(f"### Phương trình hồi quy ước lượng từ dữ liệu\n")
    md.append(f"### `{res['equation']}`\n")
    md.append("**Bảng hệ số hồi quy**\n")
    md.append("| Predictor | Coefficient | Std. Error | t-value | p-value |")
    md.append("| --- | ---: | ---: | ---: | ---: |")
    md.append(f"| Intercept (β0) | {res['b0']:.3f} | {res['se_const']:.3f} | "
              f"{res['t_const']:.3f} | {fmt_p(res['p_const'])} |")
    md.append(f"| Privacy Risk (β1) | {res['b1']:.3f} | {res['se_b1']:.3f} | "
              f"{res['t_b1']:.3f} | {fmt_p(res['p_b1'])} |")
    md.append("")
    md.append("**Chỉ số mô hình**\n")
    md.append("| Model | R² | Adjusted R² | N |")
    md.append("| --- | ---: | ---: | ---: |")
    md.append(f"| Model 1 | {res['r2']:.3f} | {res['adj_r2']:.3f} | {res['n']} |")
    md.append("")
    md.append(f"- **F-statistic:** {res['f_stat']:.3f} (p = {fmt_p(res['f_pvalue'])})")
    md.append(f"- **p-value của hệ số Privacy Risk:** {fmt_p(res['p_b1'])} — "
              f"{significance_label(res['p_b1'])}")
    md.append(f"- **Diễn giải hệ số:** khi Privacy Risk tăng 1 điểm, Support thay đổi "
              f"trung bình {res['b1']:.3f} điểm (giữ nguyên các yếu tố khác).")
    md.append(f"- **R² = {res['r2']:.3f}:** mô hình giải thích khoảng "
              f"{res['r2']*100:.1f}% phương sai của Support.")
    return "\n".join(md)


# =============================================================================
# 7. SCENARIO COMPARISON  +  8. ANOVA
# =============================================================================
def run_scenario_comparison(clean_data):
    rows = []
    for code in (1, 2, 3):
        sub = clean_data[clean_data["scenario_code"] == code]["support"]
        rows.append({
            "code": code,
            "label": SCENARIO_LABELS[code],
            "label_en": SCENARIO_LABELS_EN[code],
            "mean": sub.mean() if len(sub) else np.nan,
            "sd": sub.std(ddof=1) if len(sub) > 1 else np.nan,
            "n": len(sub),
        })
    return pd.DataFrame(rows)


def run_anova(clean_data):
    groups = [clean_data[clean_data["scenario_code"] == c]["support"].dropna().values
              for c in (1, 2, 3)]
    usable = [g for g in groups if len(g) >= 2]
    if len(usable) < 2:
        return {"ok": False, "reason": "Không đủ nhóm có dữ liệu để chạy ANOVA."}
    f_stat, p_value = stats.f_oneway(*usable)
    n_total = sum(len(g) for g in groups)
    k = len([g for g in groups if len(g) > 0])
    df_between = k - 1
    df_within = n_total - k

    # Eta squared
    grand_mean = np.concatenate([g for g in groups if len(g)]).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g))
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups if len(g))
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

    # Levene's test (kiểm tra phương sai đồng nhất)
    try:
        lev_stat, lev_p = stats.levene(*usable)
    except Exception:
        lev_stat, lev_p = np.nan, np.nan

    # Kruskal-Wallis (kiểm định phi tham số bổ sung)
    try:
        kw_stat, kw_p = stats.kruskal(*usable)
    except Exception:
        kw_stat, kw_p = np.nan, np.nan

    result = {
        "ok": True, "f_stat": f_stat, "p_value": p_value, "n": n_total,
        "df_between": df_between, "df_within": df_within, "eta_sq": eta_sq,
        "levene_stat": lev_stat, "levene_p": lev_p,
        "kw_stat": kw_stat, "kw_p": kw_p, "posthoc": None,
    }

    if p_value < 0.05:
        try:
            sub = clean_data.dropna(subset=["support"])
            tukey = pairwise_tukeyhsd(endog=sub["support"],
                                      groups=sub["scenario_en"], alpha=0.05)
            result["posthoc"] = pd.DataFrame(tukey.summary().data[1:],
                                             columns=tukey.summary().data[0])
        except Exception as e:
            result["posthoc_error"] = str(e)
    return result


def scenario_markdown(desc_df, anova_res):
    md = []
    md.append("## 📊 So sánh mức độ ủng hộ giữa các tình huống sử dụng\n")
    md.append("**Thống kê mô tả theo tình huống (Use-Case Scenario)**\n")
    md.append("| Use Case | Mean Support | SD | N |")
    md.append("| --- | ---: | ---: | ---: |")
    for _, r in desc_df.iterrows():
        mean = f"{r['mean']:.3f}" if pd.notna(r["mean"]) else "n/a"
        sd = f"{r['sd']:.3f}" if pd.notna(r["sd"]) else "n/a"
        md.append(f"| {r['label_en']}<br>*{r['label']}* | {mean} | {sd} | {int(r['n'])} |")
    md.append("")
    md.append("### Kiểm định One-way ANOVA\n")
    if not anova_res.get("ok"):
        md.append(f"*{anova_res.get('reason', 'Không chạy được ANOVA.')}*")
        return "\n".join(md)

    md.append("| Chỉ số | Giá trị |")
    md.append("| --- | ---: |")
    md.append(f"| F-statistic | **{anova_res['f_stat']:.3f}** |")
    md.append(f"| p-value | **{fmt_p(anova_res['p_value'])}** |")
    md.append(f"| Bậc tự do (giữa nhóm, trong nhóm) | ({anova_res['df_between']}, {anova_res['df_within']}) |")
    md.append(f"| Eta squared (η²) | {anova_res['eta_sq']:.3f} |")
    md.append(f"| Tổng số quan sát N | {anova_res['n']} |")
    md.append("")
    md.append(f"**Mức ý nghĩa:** {significance_label(anova_res['p_value'])}")
    md.append("")
    md.append("**Kiểm định bổ trợ**\n")
    md.append("| Kiểm định | Thống kê | p-value | Ghi chú |")
    md.append("| --- | ---: | ---: | --- |")
    md.append(f"| Levene (phương sai đồng nhất) | {anova_res['levene_stat']:.3f} | "
              f"{fmt_p(anova_res['levene_p'])} | p > 0.05 nghĩa là giả định ANOVA được thoả mãn |")
    md.append(f"| Kruskal-Wallis (phi tham số) | {anova_res['kw_stat']:.3f} | "
              f"{fmt_p(anova_res['kw_p'])} | Dùng khi dữ liệu không phân phối chuẩn |")
    md.append("")

    if anova_res.get("posthoc") is not None:
        md.append("**Kiểm định hậu định Tukey HSD** (ANOVA có ý nghĩa thống kê)\n")
        ph = anova_res["posthoc"]
        md.append("| Nhóm 1 | Nhóm 2 | Chênh lệch trung bình | p-adj | Khác biệt có ý nghĩa |")
        md.append("| --- | --- | ---: | ---: | :---: |")
        for _, r in ph.iterrows():
            reject = "Có" if str(r["reject"]).lower() in ("true", "1") else "Không"
            md.append(f"| {r['group1']} | {r['group2']} | {float(r['meandiff']):.3f} | "
                      f"{float(r['p-adj']):.3f} | {reject} |")
        md.append("")
    elif anova_res["p_value"] >= 0.05:
        md.append("*ANOVA không có ý nghĩa thống kê ở mức p < 0.05, nên không thực hiện "
                  "kiểm định hậu định (post-hoc).*")
        md.append("")

    md.append(
        "> *Giả thuyết H2 dự đoán mức độ ủng hộ khác nhau giữa các tình huống, trong đó "
        "tình huống giám sát biểu tình được kỳ vọng nhận ít ủng hộ hơn tình huống tìm người "
        "mất tích. Bảng trên chỉ trình bày số liệu; ứng dụng không tự xếp hạng tình huống nào "
        "là 'tốt nhất' hay 'xấu nhất' và không tự kết luận về giả thuyết.*"
    )
    return "\n".join(md)


# =============================================================================
# 9. VISUALIZATION
# =============================================================================
def plot_regression(clean_data, reg_res):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    palette = {SCENARIO_LABELS[1]: "#2E86AB",
               SCENARIO_LABELS[2]: "#F6A02D",
               SCENARIO_LABELS[3]: "#C7423C"}
    for label, grp in clean_data.groupby("scenario"):
        ax.scatter(grp["privacy_risk"], grp["support"], s=55, alpha=0.75,
                   edgecolor="white", linewidth=0.8,
                   color=palette.get(label, "#666666"), label=label)

    x_line = np.linspace(clean_data["privacy_risk"].min() - 0.15,
                         clean_data["privacy_risk"].max() + 0.15, 100)
    y_line = reg_res["b0"] + reg_res["b1"] * x_line
    ax.plot(x_line, y_line, color="#1B1B1B", linewidth=2.4,
            label="Đường hồi quy tuyến tính")

    ax.set_title("Mối quan hệ giữa nhận thức về rủi ro quyền riêng tư\n"
                 "và mức độ ủng hộ Facial Recognition",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Nhận thức rủi ro quyền riêng tư (Perceived Privacy Risk, 1–5)", fontsize=11)
    ax.set_ylabel("Mức độ ủng hộ cảnh sát dùng nhận diện khuôn mặt (1–5)", fontsize=11)

    p_txt = fmt_p(reg_res["p_b1"])
    p_txt = p_txt if p_txt.startswith("<") else "= " + p_txt
    textbox = (f"{reg_res['equation']}\n"
               f"R² = {reg_res['r2']:.3f}   |   "
               f"p {p_txt}   |   N = {reg_res['n']}")
    ax.text(0.03, 0.04, textbox, transform=ax.transAxes, fontsize=10.5,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFDF5",
                      edgecolor="#999999", alpha=0.95))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, title="Tình huống")
    ax.set_ylim(0.5, 5.5)
    fig.tight_layout()
    return fig


def plot_scenario_bars(desc_df, clean_data):
    import textwrap
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = ["#2E86AB", "#F6A02D", "#C7423C"]
    labels = [textwrap.fill(r["label"], 20) + f"\n(N = {int(r['n'])})"
              for _, r in desc_df.iterrows()]
    means = desc_df["mean"].values
    sds = desc_df["sd"].fillna(0).values

    bars = ax.bar(labels, means, yerr=sds, capsize=7, color=colors,
                  alpha=0.9, edgecolor="#333333", linewidth=0.8,
                  error_kw=dict(ecolor="#555555", lw=1.4))

    # Chồng các điểm dữ liệu cá nhân
    for i, code in enumerate((1, 2, 3)):
        vals = clean_data[clean_data["scenario_code"] == code]["support"].values
        if len(vals):
            jitter = np.random.default_rng(42).normal(0, 0.055, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=22,
                       color="#222222", alpha=0.35, zorder=3)

    for bar, m, sd in zip(bars, means, sds):
        if pd.notna(m):
            ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.15, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=11.5, fontweight="bold")

    ax.set_title("Mức độ ủng hộ trung bình theo từng tình huống sử dụng",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Mức độ ủng hộ trung bình (Mean Support, 1–5)", fontsize=11)
    ax.set_xlabel("Tình huống sử dụng (Use-Case Scenario)", fontsize=11)
    ax.set_ylim(0, 6.0)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    fig.tight_layout()
    return fig


# =============================================================================
# TRÌNH BÀY PHẦN TỔNG QUAN & BIẾN NGHIÊN CỨU
# =============================================================================
def overview_markdown(summary, file_path):
    md = []
    md.append("## 🧹 Tổng quan dữ liệu và làm sạch dữ liệu\n")
    md.append(SAMPLE_WARNING + "\n")
    md.append(f"*Nguồn dữ liệu:* `{os.path.basename(file_path)}`\n")
    md.append("| Bước | Số lượng |")
    md.append("| --- | ---: |")
    md.append(f"| Tổng số respondent ban đầu | **{summary['n_initial']}** |")
    md.append(f"| Bị loại vì không đồng ý tham gia (consent) | {summary['n_dropped_consent']} |")
    md.append(f"| Bị loại vì không vượt qua attention check | {summary['n_dropped_attention']} |")
    md.append(f"| Bị loại vì không xác định được scenario (thiếu tháng sinh) | {summary['n_dropped_scenario']} |")
    md.append(f"| Bị loại vì thiếu dữ liệu privacy_risk / support | {summary['n_dropped_missing']} |")
    md.append(f"| **Số respondent còn lại sau data cleaning** | **{summary['n_clean']}** |")
    md.append("")
    acc = summary["manip_accuracy"]
    acc_txt = f"{acc*100:.1f}%" if pd.notna(acc) else "n/a"
    md.append("**Kiểm tra chất lượng dữ liệu**\n")
    md.append("| Chỉ số | Giá trị |")
    md.append("| --- | ---: |")
    md.append(f"| Tỷ lệ trả lời đúng manipulation check | **{acc_txt}** "
              f"({summary['n_manip_correct']}/{summary['n_clean']}) |")
    alpha = summary["alpha_privacy"]
    alpha_txt = f"{alpha:.3f}" if pd.notna(alpha) else "n/a"
    md.append(f"| Cronbach's α của thang Privacy Risk | {alpha_txt} |")
    md.append("")
    md.append("> *Theo thiết kế mặc định, respondent trả lời sai manipulation check "
              "**vẫn được giữ lại** trong phân tích chính; tỷ lệ trả lời đúng được báo cáo "
              "riêng như một chỉ báo chất lượng dữ liệu. Bạn có thể bật tuỳ chọn loại bỏ "
              "nhóm này ở phần cài đặt phía trên.*")
    return "\n".join(md)


def variables_markdown(clean_data, summary):
    md = []
    md.append("## 🧮 Các biến nghiên cứu\n")
    md.append("| Tên biến | Loại | Mô tả | Cách xây dựng |")
    md.append("| --- | --- | --- | --- |")
    md.append("| `privacy_risk` | Biến độc lập 1 (liên tục, 1–5) | "
              "Nhận thức về rủi ro quyền riêng tư — mức độ người trả lời xem nhận diện "
              "khuôn mặt là mối đe doạ đối với quyền riêng tư | "
              f"Trung bình của {len(summary['privacy_items'])} item Likert |")
    md.append("| `support` | Biến phụ thuộc (liên tục, 1–5) | "
              "Mức độ ủng hộ việc cảnh sát sử dụng nhận diện khuôn mặt | "
              "Trung bình 3 item Likert trong đúng block scenario của respondent |")
    md.append("| `scenario` | Biến độc lập 2 (phân loại, 3 nhóm) | "
              "Tình huống sử dụng mà respondent được đọc | "
              "Xác định từ tháng sinh (Jan/Apr/Jul/Oct → tìm người mất tích; "
              "Feb/May/Aug/Nov → phòng ngừa tội phạm; Mar/Jun/Sep/Dec → giám sát biểu tình) |")
    md.append("| `manipulation_check` | Biến kiểm tra (nhị phân 0/1) | "
              "Respondent có nhớ đúng tình huống đã đọc hay không | "
              "So sánh câu trả lời với scenario thực tế (1 = đúng, 0 = sai) |")
    md.append("")
    md.append("**Các item cấu thành thang Privacy Risk:**\n")
    for i, item in enumerate(summary["privacy_items"], 1):
        md.append(f"{i}. {item}")
    md.append("")
    md.append("**Thống kê mô tả các biến liên tục**\n")
    md.append("| Biến | N | Mean | SD | Min | Max |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for var, name in [("privacy_risk", "privacy_risk"), ("support", "support")]:
        s = clean_data[var]
        md.append(f"| {name} | {s.count()} | {s.mean():.3f} | {s.std(ddof=1):.3f} | "
                  f"{s.min():.2f} | {s.max():.2f} |")
    md.append("")
    md.append("**Phân bố respondent theo scenario**\n")
    md.append("| Tình huống | N |")
    md.append("| --- | ---: |")
    for code in (1, 2, 3):
        n = int((clean_data["scenario_code"] == code).sum())
        md.append(f"| {SCENARIO_LABELS[code]} | {n} |")
    md.append("")
    md.append("> *Lưu ý phương pháp: mô hình chính chỉ gồm Perceived Privacy Risk, Support và "
              "Use-Case Scenario. Các biến nhân khẩu học (tuổi, giới tính, nghề nghiệp, mức độ "
              "quen thuộc) chỉ dùng cho mô tả, không đưa vào mô hình hồi quy. Dữ liệu NIST là "
              "nguồn bên ngoài dùng cho phần thảo luận, không phải biến trong hồi quy này.*")
    return "\n".join(md)


def hypotheses_markdown(corr_res, reg_res, desc_df, anova_res):
    md = []
    md.append("## 🎯 Tóm tắt kết quả theo hai giả thuyết\n")
    md.append("### H1 — Privacy Risk → Support\n")
    md.append("*Higher perceived privacy risk is negatively associated with support for "
              "police use of facial recognition in public spaces.*\n")
    md.append("| Bằng chứng thống kê | Giá trị |")
    md.append("| --- | ---: |")
    md.append(f"| Pearson r | {corr_res['r']:.3f} |")
    md.append(f"| p-value (tương quan) | {fmt_p(corr_res['p'])} |")
    md.append(f"| Hệ số hồi quy β1 | {reg_res['b1']:.3f} |")
    md.append(f"| p-value (β1) | {fmt_p(reg_res['p_b1'])} |")
    md.append(f"| R² | {reg_res['r2']:.3f} |")
    md.append(f"| N | {reg_res['n']} |")
    md.append("")
    md.append("### H2 — Scenario → Support\n")
    md.append("*Public support for police facial recognition differs across use cases, with "
              "protest/political-monitoring use expected to receive lower support than "
              "missing-person identification.*\n")
    md.append("| Bằng chứng thống kê | Giá trị |")
    md.append("| --- | ---: |")
    if anova_res.get("ok"):
        md.append(f"| One-way ANOVA F | {anova_res['f_stat']:.3f} |")
        md.append(f"| p-value | {fmt_p(anova_res['p_value'])} |")
        md.append(f"| η² | {anova_res['eta_sq']:.3f} |")
        md.append(f"| N | {anova_res['n']} |")
    else:
        md.append("| ANOVA | Không chạy được |")
    for _, r in desc_df.iterrows():
        mean = f"{r['mean']:.3f}" if pd.notna(r["mean"]) else "n/a"
        md.append(f"| Mean Support — {r['label']} | {mean} |")
    md.append("")
    md.append("> *Các con số trên được tính trực tiếp từ dữ liệu. Việc kết luận giả thuyết "
              "được ủng hộ hay không thuộc về người phân tích, không phải ứng dụng.*")
    return "\n".join(md)


def demographics_markdown(clean_data, cols):
    md = ["## 👥 Mô tả mẫu (chỉ mang tính mô tả, không đưa vào mô hình)\n"]
    any_col = False
    for key, title in [("age", "Nhóm tuổi"), ("gender", "Giới tính"),
                       ("occupation", "Nghề nghiệp"),
                       ("familiarity", "Mức độ quen thuộc với công nghệ (1–5)")]:
        col = cols.get(key)
        if col is None or col not in clean_data.columns:
            continue
        any_col = True
        counts = clean_data[col].value_counts(dropna=False)
        md.append(f"**{title}**\n")
        md.append("| Giá trị | N | % |")
        md.append("| --- | ---: | ---: |")
        for val, n in counts.items():
            pct = n / len(clean_data) * 100
            md.append(f"| {val} | {n} | {pct:.1f}% |")
        md.append("")
    if not any_col:
        md.append("*Không tìm thấy cột nhân khẩu học trong dữ liệu.*")
    return "\n".join(md)



# =============================================================================
# 10a. STATE + HÀM XỬ LÝ (giữ nguyên từ bản Gradio)
# =============================================================================
STATE = {"raw": None, "path": None}


def load_into_state(file_obj):
    path = None
    if file_obj is not None:
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    df_raw, used_path = load_excel(path)
    STATE["raw"] = df_raw
    STATE["path"] = used_path
    return df_raw, used_path


def reload_data(file_obj):
    """Nút 'Tải lại dữ liệu'."""
    try:
        df_raw, used_path = load_into_state(file_obj)
        return (f"✅ Đã đọc lại dữ liệu từ `{os.path.basename(used_path)}` — "
                f"{len(df_raw)} dòng, {len(df_raw.columns)} cột.\n\n{SAMPLE_WARNING}")
    except Exception as e:
        return f"❌ Lỗi khi đọc file: {e}"


def analyze(file_obj, exclude_failed_manip):
    """Nút 'Phân tích dữ liệu' — chạy toàn bộ quy trình phân tích."""
    try:
        df_raw, used_path = load_into_state(file_obj)
    except Exception as e:
        err = f"❌ **Lỗi khi đọc file Excel:** {e}"
        return err, "", "", "", "", "", "", None, None

    cols = detect_columns(df_raw)
    if cols["missing"]:
        err = ("❌ **Không nhận diện được cấu trúc file khảo sát.** Thiếu: "
               + ", ".join(cols["missing"])
               + "\n\nHãy kiểm tra xem file Excel có đúng là bản export của form "
                 "'Public Attitudes Toward Facial Recognition Technology' hay không.")
        return err, "", "", "", "", "", "", None, None

    try:
        clean_data, summary = clean_and_build(df_raw, cols)
    except Exception as e:
        return f"❌ **Lỗi khi làm sạch dữ liệu:** {e}", "", "", "", "", "", "", None, None

    note = ""
    if exclude_failed_manip:
        n_before = len(clean_data)
        clean_data = clean_data[clean_data["manipulation_check"] == 1].reset_index(drop=True)
        note = (f"\n\n🔎 *Đang áp dụng tuỳ chọn: **loại bỏ** {n_before - len(clean_data)} "
                f"respondent trả lời sai manipulation check. Còn lại {len(clean_data)} "
                f"respondent trong phân tích bên dưới.*")

    if len(clean_data) < 3:
        return (overview_markdown(summary, used_path) + note
                + "\n\n❌ Không đủ dữ liệu hợp lệ để chạy phân tích thống kê.",
                "", "", "", "", "", "", None, None)

    # --- Chạy các phân tích --------------------------------------------------
    correlation_result = run_correlation(clean_data)
    regression_result = run_regression(clean_data)
    desc_df = run_scenario_comparison(clean_data)
    anova_result = run_anova(clean_data)

    overview_md = overview_markdown(summary, used_path) + note
    variables_md = variables_markdown(clean_data, summary)
    corr_md = correlation_markdown(correlation_result)
    reg_md = regression_markdown(regression_result)
    scen_md = scenario_markdown(desc_df, anova_result)
    hypo_md = hypotheses_markdown(correlation_result, regression_result,
                                  desc_df, anova_result)
    demo_md = demographics_markdown(clean_data, cols)

    fig_reg = plot_regression(clean_data, regression_result)
    fig_scen = plot_scenario_bars(desc_df, clean_data)

    status = (f"✅ **Phân tích hoàn tất** — {summary['n_initial']} dòng gốc → "
              f"{len(clean_data)} respondent được phân tích.\n\n{SAMPLE_WARNING}")

    return (status, overview_md, variables_md, corr_md, reg_md, scen_md, hypo_md,
            fig_reg, fig_scen)


# =============================================================================
# 10b. STREAMLIT INTERFACE
# =============================================================================
INITIAL_STATUS = "*Nhấn **Phân tích dữ liệu** để bắt đầu.*"
EMPTY_RESULT = (INITIAL_STATUS, "", "", "", "", "", "", None, None)


def save_upload(uploaded):
    """Lưu file người dùng tải lên ra đĩa (giữ nguyên tên) và trả về đường dẫn.
    Trả về None nếu chưa có file -> ứng dụng tự dò tìm file mặc định."""
    if uploaded is None:
        return None
    if "tmp_dir" not in st.session_state:
        st.session_state["tmp_dir"] = tempfile.mkdtemp(prefix="survey_")
    path = os.path.join(st.session_state["tmp_dir"], os.path.basename(uploaded.name))
    with open(path, "wb") as f:
        f.write(uploaded.getvalue())
    return path


def on_analyze():
    """Callback của nút 'Phân tích dữ liệu' và của checkbox (giống Gradio:
    đổi checkbox sẽ chạy lại phân tích)."""
    path = save_upload(st.session_state.get("file_input"))
    result = analyze(path, st.session_state.get("exclude_manip", False))
    # Đóng figure trong pyplot để không rò rỉ bộ nhớ (vẫn hiển thị được qua st.pyplot)
    for fig in result[7:9]:
        if fig is not None:
            plt.close(fig)
    st.session_state["result"] = result


def on_reload():
    """Callback của nút 'Tải lại dữ liệu': chỉ cập nhật dòng trạng thái."""
    path = save_upload(st.session_state.get("file_input"))
    msg = reload_data(path)
    prev = st.session_state.get("result", EMPTY_RESULT)
    st.session_state["result"] = (msg,) + tuple(prev[1:])


def build_interface():
    st.title("🎭 Phân tích thái độ đối với công nghệ nhận diện khuôn mặt")
    st.markdown(
        "**Ứng dụng phân tích dữ liệu khảo sát về nhận thức rủi ro quyền riêng tư, "
        "mức độ ủng hộ việc cảnh sát sử dụng công nghệ nhận diện khuôn mặt và sự khác "
        "biệt giữa các tình huống sử dụng.**\n\n"
        "*Dự án môn Culture and Coding: Python — Python được dùng để làm sạch dữ liệu, "
        "tạo biến nghiên cứu và kiểm chứng hai giả thuyết H1 và H2.*"
    )
    st.markdown(f"### {SAMPLE_WARNING}")

    if "result" not in st.session_state:
        st.session_state["result"] = EMPTY_RESULT

    with st.expander("📂 Nguồn dữ liệu và tuỳ chọn phân tích", expanded=True):
        col_file, col_opt = st.columns([3, 2])
        with col_file:
            st.file_uploader(
                "Chọn file Excel khảo sát (.xlsx) — bỏ trống để ứng dụng tự dò tìm",
                type=["xlsx", "xls"], key="file_input",
            )
        with col_opt:
            st.checkbox(
                "Loại bỏ respondent trả lời sai manipulation check",
                value=False, key="exclude_manip", on_change=on_analyze,
                help="Mặc định: GIỮ LẠI để phân tích chính và báo cáo tỷ lệ đúng riêng.",
            )
        col_a, col_r = st.columns([3, 1])
        with col_a:
            st.button("▶️ Phân tích dữ liệu", type="primary",
                      use_container_width=True, on_click=on_analyze)
        with col_r:
            st.button("🔄 Tải lại dữ liệu", use_container_width=True, on_click=on_reload)

    (status, overview, variables, corr, reg, scen, hypo,
     fig_reg, fig_scen) = st.session_state["result"]

    st.markdown(status)

    tabs = st.tabs([
        "A. Tổng quan dữ liệu", "B. Biến nghiên cứu", "C. Tương quan",
        "D. Hồi quy tuyến tính", "E. So sánh tình huống", "F. Tóm tắt giả thuyết",
    ])
    with tabs[0]:
        st.markdown(overview)
    with tabs[1]:
        st.markdown(variables)
    with tabs[2]:
        st.markdown(corr)
    with tabs[3]:
        st.markdown(reg)
        if fig_reg is not None:
            st.pyplot(fig_reg)
    with tabs[4]:
        # unsafe_allow_html để hiển thị thẻ <br> trong bảng thống kê mô tả
        st.markdown(scen, unsafe_allow_html=True)
        if fig_scen is not None:
            st.pyplot(fig_scen)
    with tabs[5]:
        st.markdown(hypo)

    st.markdown(
        "---\n"
        "**Phương pháp:** H1 kiểm định bằng tương quan Pearson và hồi quy tuyến tính đơn "
        "(`support ~ privacy_risk`). H2 kiểm định bằng One-way ANOVA so sánh `support` "
        "giữa ba tình huống, kèm kiểm định hậu định Tukey HSD khi ANOVA có ý nghĩa.\n\n"
        "*Ứng dụng trình bày kết quả thống kê để người đọc tự đánh giá; không tự kết luận "
        "giả thuyết được chấp nhận hay bác bỏ.*"
    )


# =============================================================================
# 11. LAUNCH APPLICATION  (chạy bằng:  streamlit run app.py)
# =============================================================================
build_interface()
