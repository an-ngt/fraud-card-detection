import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# CẤU HÌNH TRANG & GIAO DIỆN
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud DSS | Hệ thống hỗ trợ quyết định",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0B3D91"      # navy - tin cậy, định chế tài chính
ACCENT = "#D7263D"       # đỏ cảnh báo
ACCENT_GOOD = "#1B998B"  # xanh ngọc - an toàn / tiết kiệm
INK = "#10131A"
PAPER = "#F6F5F1"
MUTED = "#5B6472"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Inter', sans-serif;
    color: {INK};
}}
h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}
.stApp {{
    background-color: {PAPER};
}}
section[data-testid="stSidebar"] {
    background-color: {INK};
}
section[data-testid="stSidebar"] * {
    color: #E9E9E6 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2A2F3A;
}

/* Thẻ KPI */
.kpi-card {{
    background: white;
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid #E4E2DA;
    border-left: 5px solid {PRIMARY};
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.kpi-card.alert {{ border-left-color: {ACCENT}; }}
.kpi-card.good {{ border-left-color: {ACCENT_GOOD}; }}
.kpi-label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {MUTED};
    font-weight: 600;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: {INK};
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 0.80rem;
    color: {MUTED};
    margin-top: 4px;
}}

/* Banner khuyến nghị */
.reco-banner {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #15205C 100%);
    color: white;
    border-radius: 12px;
    padding: 22px 26px;
    margin: 6px 0 18px 0;
}}
.reco-banner h3 {{ color: white !important; margin: 0 0 6px 0; }}
.reco-banner p {{ color: #D9E0F5; margin: 0; font-size: 0.95rem; }}

.section-eyebrow {{
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
    font-weight: 700;
    margin-bottom: -6px;
}}

.risk-pill {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}}
.note-box {{
    background: #FFF8E6;
    border: 1px solid #F0D793;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #6B5413;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DEFAULT_MODEL_PATH = Path(__file__).parent / "fraud_lightgbm_dss.pkl"

CANDIDATE_AMOUNT_COLS = ["amount", "amt", "transaction_amount", "txn_amount", "so_tien", "amount_usd"]
CANDIDATE_LABEL_COLS = ["is_fraud", "label", "fraud", "class", "Class", "target", "is_fraud_actual"]
CANDIDATE_MERCHANT_COLS = ["merchant", "merchant_name", "merchant_category", "category"]
CANDIDATE_COUNTRY_COLS = ["country", "region", "khu_vuc", "quoc_gia"]
CANDIDATE_ID_COLS = ["transaction_id", "txn_id", "id"]


# ──────────────────────────────────────────────────────────────────────────
# TẢI MÔ HÌNH
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_bundle(file_bytes_or_path):
    if isinstance(file_bytes_or_path, (str, Path)):
        with open(file_bytes_or_path, "rb") as f:
            bundle = pickle.load(f)
    else:
        bundle = pickle.load(io.BytesIO(file_bytes_or_path))
    return bundle


def get_pandas_categorical(model):
    """Lấy danh sách categories đã dùng lúc huấn luyện, theo đúng thứ tự cột categorical."""
    try:
        return model.booster_.pandas_categorical
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# DỮ LIỆU MẪU (DEMO) — dùng khi người dùng chưa có file để thử nghiệm hệ thống
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_demo_data(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    df["transaction_id"] = [f"TXN{100000+i}" for i in range(n)]

    hour = rng.integers(0, 24, n)
    df["hour"] = hour
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)
    df["is_online"] = rng.integers(0, 2, n)
    df["age"] = rng.integers(18, 75, n)
    df["gender"] = rng.choice(["F", "M"], n)

    amount = np.round(np.exp(rng.normal(3.4, 1.25, n)), 2)
    df["amount"] = amount
    df["amt_log"] = np.log1p(amount)
    df["customer_avg_spending"] = np.round(amount * rng.uniform(0.6, 1.6, n), 2)
    df["amt_ratio"] = df["amount"] / df["customer_avg_spending"].replace(0, 1)
    df["distance_km"] = np.round(rng.exponential(15, n), 2)
    df["time_since_last_txn"] = np.round(rng.exponential(12, n), 2)
    df["txn_count_24h"] = rng.poisson(3, n)

    df["merchant"] = rng.choice(
        ["Điện máy", "Siêu thị", "Nhà hàng", "Du lịch", "Thời trang", "Cửa hàng trực tuyến", "Xăng dầu", "Y tế"], n
    )
    df["merchant_freq"] = np.round(rng.uniform(0, 1, n), 3)
    df["country"] = rng.choice(
        ["Việt Nam", "Singapore", "Mỹ", "Nhật Bản", "Anh", "Hàn Quốc"], n, p=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06]
    )
    df["category_risk_tier"] = rng.choice(["Low", "Medium", "High"], n, p=[0.5, 0.35, 0.15])
    df["is_night_x_online"] = df["is_night"] * df["is_online"]
    df["amt_log_x_is_night"] = df["amt_log"] * df["is_night"]

    risk_score = (
        0.9 * (df["amt_ratio"] > 3).astype(int)
        + 1.1 * df["is_night"]
        + 0.4 * (df["category_risk_tier"] == "High").astype(int)
        + 0.3 * (df["country"] != "Việt Nam").astype(int)
        + 0.5 * (df["distance_km"] > 50).astype(int)
        + rng.normal(0, 0.3, n)
    )
    p_true = 1 / (1 + np.exp(-(risk_score - 3)))
    df["is_fraud"] = (rng.uniform(0, 1, n) < p_true * 0.18).astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────
# TIỀN XỬ LÝ / KỸ THUẬT ĐẶC TRƯNG
# ──────────────────────────────────────────────────────────────────────────
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_features(df_raw, feature_cols, categorical_features, pandas_categorical):
    """
    Tái tạo toàn bộ pipeline tiền xử lý từ dữ liệu thô theo đúng notebook DA1_EDA.ipynb.
    Nếu một cột đã tồn tại sẵn thì bỏ qua bước tính lại để tránh ghi đè.
    """
    df = df_raw.copy()
    notes = []

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 1 — Ép kiểu dữ liệu (Cell 1.2)
    # ══════════════════════════════════════════════════════════════════
    for dt_col in ["trans_date_trans_time"]:
        if dt_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            notes.append(f"Đã chuyển `{dt_col}` sang datetime.")

    if "dob" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["dob"]):
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # cc_num dùng làm key groupby — đảm bảo là string
    if "cc_num" in df.columns:
        df["cc_num"] = df["cc_num"].astype(str)

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 2 — Loại bỏ cột thừa (Cell 5)
    # ══════════════════════════════════════════════════════════════════
    for drop_col in ["merch_zipcode", "Unnamed: 0", "trans_num"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 3 — amt_log: log(1 + amt)  (Cell 6)
    # ══════════════════════════════════════════════════════════════════
    if "amt_log" not in df.columns:
        amt_col_raw = find_col(df, CANDIDATE_AMOUNT_COLS)
        if amt_col_raw:
            df["amt_log"] = np.log1p(df[amt_col_raw].clip(lower=0))
            notes.append(f"Đã tính `amt_log = log(1 + {amt_col_raw})`.")
        else:
            df["amt_log"] = 0.0
            notes.append("⚠️ Không tìm thấy cột số tiền — `amt_log` gán 0.")

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 4 — distance_km bằng Haversine  (Cell 7)
    # ══════════════════════════════════════════════════════════════════
    if "distance_km" not in df.columns:
        geo_cols = {"lat", "long", "merch_lat", "merch_long"}
        if geo_cols.issubset(df.columns):
            lon1 = np.radians(df["long"].astype(float))
            lat1 = np.radians(df["lat"].astype(float))
            lon2 = np.radians(df["merch_long"].astype(float))
            lat2 = np.radians(df["merch_lat"].astype(float))
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            df["distance_km"] = 6367.0 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
            # Xóa 4 cột tọa độ gốc như notebook
            df = df.drop(columns=[c for c in geo_cols if c in df.columns])
            notes.append("Đã tính `distance_km` (Haversine) và xóa 4 cột tọa độ gốc.")
        else:
            df["distance_km"] = 0.0
            notes.append("⚠️ Thiếu cột lat/long — `distance_km` gán 0.")

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 5 — Trích xuất đặc trưng thời gian  (Cell 8)
    # ══════════════════════════════════════════════════════════════════
    # trans_hour (cột trung gian, sẽ bị xóa sau bước Circular Encoding)
    if "trans_hour" not in df.columns:
        if "trans_date_trans_time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["trans_date_trans_time"]):
            df["trans_hour"] = df["trans_date_trans_time"].dt.hour
        elif "unix_time" in df.columns:
            df["trans_hour"] = pd.to_datetime(df["unix_time"], unit="s").dt.hour
            notes.append("Đã trích xuất `trans_hour` từ `unix_time`.")

    # age = năm giao dịch - năm sinh (Cell 8, công thức đúng của notebook)
    if "age" not in df.columns and "dob" in df.columns and "trans_date_trans_time" in df.columns:
        df["age"] = (
            df["trans_date_trans_time"].dt.year - df["dob"].dt.year
        ).clip(0, 120).fillna(df["dob"].apply(lambda x: pd.Timestamp.now().year - x.year if pd.notna(x) else np.nan).median())
        notes.append("Đã tính `age` = năm_GD - năm_sinh.")

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 6 — Circular Encoding & is_night & is_online  (Cell 10)
    # ══════════════════════════════════════════════════════════════════
    if "sin_hour" not in df.columns or "cos_hour" not in df.columns:
        if "trans_hour" in df.columns:
            df["sin_hour"] = np.sin(2 * np.pi * df["trans_hour"] / 24)
            df["cos_hour"] = np.cos(2 * np.pi * df["trans_hour"] / 24)
            notes.append("Đã tính `sin_hour`/`cos_hour` (Circular Encoding).")

    # is_night: 22h–3h (định nghĩa trong Cell 10, cập nhật lại từ Cell 8)
    if "is_night" not in df.columns:
        if "trans_hour" in df.columns:
            df["is_night"] = ((df["trans_hour"] >= 22) | (df["trans_hour"] <= 3)).astype(int)
            notes.append("Đã tính `is_night` (22h–3h sáng = ban đêm).")
        elif "sin_hour" in df.columns and "cos_hour" in df.columns:
            hr = (np.degrees(np.arctan2(df["sin_hour"], df["cos_hour"])) / 15.0) % 24
            df["is_night"] = ((hr >= 22) | (hr <= 3)).astype(int)
            notes.append("Đã suy ra `is_night` từ `sin_hour`/`cos_hour`.")

    # is_online: category chứa chữ 'net' (Cell 10)
    if "is_online" not in df.columns:
        cat_col_raw = find_col(df, ["category"])
        if cat_col_raw:
            df["is_online"] = df[cat_col_raw].astype(str).str.contains("net", case=False, na=False).astype(int)
            notes.append("Đã tính `is_online` từ `category` (chứa 'net' = trực tuyến).")
        else:
            df["is_online"] = 0
            notes.append("⚠️ Không có `category` — `is_online` gán 0.")

    # Xóa trans_hour sau khi đã dùng xong (như notebook Cell 10)
    if "trans_hour" in df.columns:
        df = df.drop(columns=["trans_hour"])

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 7 — Đặc trưng tương tác EDA  (Cell 26)
    # ══════════════════════════════════════════════════════════════════

    # is_night_x_online
    if "is_night_x_online" not in df.columns and {"is_night", "is_online"}.issubset(df.columns):
        df["is_night_x_online"] = df["is_night"] * df["is_online"]

    # category_risk_tier — mapping CHÍNH XÁC từ notebook
    if "category_risk_tier" not in df.columns:
        cat_col_raw = find_col(df, ["category"])
        HIGH  = {"shopping_net", "misc_net", "grocery_pos", "shopping_pos"}
        MED   = {"gas_transport", "personal_care", "travel", "kids_pets", "entertainment", "misc_pos"}
        # Low = tất cả còn lại: grocery_net, food_dining, health_fitness, home

        def _assign_tier(cat):
            c = str(cat)
            if c in HIGH:
                return "High"
            elif c in MED:
                return "Medium"
            return "Low"

        if cat_col_raw:
            df["category_risk_tier"] = df[cat_col_raw].apply(_assign_tier)
            notes.append("Đã phân nhóm `category_risk_tier` (High/Medium/Low) từ `category`.")
        else:
            df["category_risk_tier"] = "Medium"
            notes.append("⚠️ Thiếu `category` — `category_risk_tier` gán Medium.")

    # amt_log_x_is_night
    if "amt_log_x_is_night" not in df.columns and {"amt_log", "is_night"}.issubset(df.columns):
        df["amt_log_x_is_night"] = df["amt_log"] * df["is_night"]

    # merchant_freq — count tuyệt đối (Cell 26)
    if "merchant_freq" not in df.columns:
        merch_col_raw = find_col(df, ["merchant"])
        if merch_col_raw:
            freq_map = df[merch_col_raw].value_counts()
            df["merchant_freq"] = df[merch_col_raw].map(freq_map).fillna(1)
            notes.append("Đã tính `merchant_freq` (số lần xuất hiện của mỗi merchant).")
        else:
            df["merchant_freq"] = 1
            notes.append("⚠️ Thiếu `merchant` — `merchant_freq` gán 1.")

    # gender mặc định
    if "gender" not in df.columns:
        df["gender"] = "F"
        notes.append("⚠️ Thiếu `gender` — gán mặc định F.")

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 8 — Đặc trưng hành vi lịch sử (expanding window, Cell 24_v2)
    # ══════════════════════════════════════════════════════════════════
    BEHAV_COLS = ["customer_total_spending", "customer_avg_spending",
                  "time_since_last_txn", "txn_count_24h", "amt_ratio"]
    need_behav = any(c not in df.columns for c in BEHAV_COLS)

    if need_behav and "cc_num" in df.columns and "unix_time" in df.columns:
        # Sắp xếp đúng như notebook
        df = df.sort_values(["cc_num", "unix_time"]).reset_index(drop=True)
        amt_col_raw = find_col(df, CANDIDATE_AMOUNT_COLS)

        if amt_col_raw:
            if "customer_total_spending" not in df.columns:
                df["customer_total_spending"] = df.groupby("cc_num")[amt_col_raw].cumsum()
                df["customer_total_spending"] = (
                    df.groupby("cc_num")["customer_total_spending"].shift(1).fillna(0)
                )

            if "customer_avg_spending" not in df.columns:
                df["customer_avg_spending"] = df.groupby("cc_num")[amt_col_raw].transform(
                    lambda x: x.expanding().mean()
                )
                df["customer_avg_spending"] = (
                    df.groupby("cc_num")["customer_avg_spending"].shift(1).fillna(0)
                )

            if "amt_ratio" not in df.columns:
                df["amt_ratio"] = np.where(
                    df["customer_avg_spending"] > 0,
                    df[amt_col_raw] / df["customer_avg_spending"],
                    1.0,
                )

        if "time_since_last_txn" not in df.columns:
            df["time_since_last_txn"] = (
                df.groupby("cc_num")["unix_time"].diff().fillna(-1)
            )

        if "txn_count_24h" not in df.columns:
            try:
                df["_dt_tmp"] = pd.to_datetime(df["unix_time"], unit="s")
                df["txn_count_24h"] = (
                    df.set_index("_dt_tmp")
                    .groupby("cc_num")[amt_col_raw if amt_col_raw else "unix_time"]
                    .rolling("24h", closed="left")
                    .count()
                    .reset_index(level=0, drop=True)
                    .values
                )
                df["txn_count_24h"] = df["txn_count_24h"].fillna(0).astype(int)
                df = df.drop(columns=["_dt_tmp"])
                notes.append("Đã tính đặc trưng hành vi lịch sử (expanding window, anti-leakage).")
            except Exception as e:
                df["txn_count_24h"] = 0
                df = df.drop(columns=["_dt_tmp"], errors="ignore")
                notes.append(f"⚠️ Không thể tính `txn_count_24h` ({e}) — gán 0.")
    elif need_behav:
        missing_prereq = []
        if "cc_num" not in df.columns:
            missing_prereq.append("`cc_num`")
        if "unix_time" not in df.columns:
            missing_prereq.append("`unix_time`")
        for c in BEHAV_COLS:
            if c not in df.columns:
                df[c] = 0 if c != "amt_ratio" else 1.0
        notes.append(
            f"⚠️ Thiếu {', '.join(missing_prereq)} — không thể tính đặc trưng hành vi lịch sử. Gán giá trị mặc định."
        )

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 9 — Gán 0 cho các cột model cần nhưng vẫn thiếu
    # ══════════════════════════════════════════════════════════════════
    still_missing = [c for c in feature_cols if c not in df.columns]
    for c in still_missing:
        df[c] = 0
        notes.append(f"⚠️ Vẫn thiếu `{c}` sau toàn bộ pipeline — gán 0.")

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 10 — Tạo ma trận X với đúng kiểu categorical
    # ══════════════════════════════════════════════════════════════════
    X = df[feature_cols].copy()

    if pandas_categorical is not None:
        cat_map = dict(zip(categorical_features, pandas_categorical))
        for col, cats in cat_map.items():
            if col not in X.columns:
                continue
            cats = list(cats)
            X[col] = X[col].astype(str)
            X[col] = X[col].where(X[col].isin(cats), cats[0])
            X[col] = pd.Categorical(X[col], categories=cats)
    else:
        for col in categorical_features:
            if col in X.columns:
                X[col] = X[col].astype("category")

    return X, df, notes


def reconstruct_hour(df):
    if "hour" in df.columns:
        return df["hour"]
    if {"sin_hour", "cos_hour"}.issubset(df.columns):
        return (np.degrees(np.arctan2(df["sin_hour"], df["cos_hour"])) / 15.0) % 24
    return pd.Series(np.nan, index=df.index)


def assign_decision(prob, threshold):
    very_high = min(0.97, threshold * 2.2)
    high = threshold
    medium = threshold * 0.5
    conditions = [prob >= very_high, prob >= high, prob >= medium]
    risk_labels = np.select(conditions, ["Very High", "High", "Medium"], default="Low")
    decision_map = {"Very High": "Chặn / Từ chối", "High": "Kiểm tra ngay", "Medium": "Theo dõi", "Low": "Phê duyệt"}
    decisions = np.array([decision_map[r] for r in risk_labels])
    return risk_labels, decisions


RISK_COLOR = {"Very High": "#D7263D", "High": "#F2994A", "Medium": "#F2C94C", "Low": "#1B998B"}


def risk_pill(label):
    color = RISK_COLOR.get(label, "#999")
    return f'<span class="risk-pill" style="background:{color}22;color:{color};border:1px solid {color}55;">{label}</span>'


# ──────────────────────────────────────────────────────────────────────────
# TÍNH CHI PHÍ THEO NGƯỠNG
# ──────────────────────────────────────────────────────────────────────────
def cost_at_threshold(prob, label, amount, threshold, cost_fp):
    pred = (prob >= threshold).astype(int)
    fp_mask = (pred == 1) & (label == 0)
    fn_mask = (pred == 0) & (label == 1)
    tp_mask = (pred == 1) & (label == 1)
    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())
    tp_count = int(tp_mask.sum())
    fn_amount = float(amount[fn_mask].sum())
    total_cost = fp_count * cost_fp + fn_amount
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else np.nan
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else np.nan
    fraud_amount_total = float(amount[label == 1].sum())
    dollar_recall = (
        100 * (fraud_amount_total - fn_amount) / fraud_amount_total if fraud_amount_total > 0 else np.nan
    )
    return {
        "threshold": threshold,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "tp_count": tp_count,
        "fn_amount": fn_amount,
        "total_cost": total_cost,
        "precision": precision,
        "recall": recall,
        "dollar_recall": dollar_recall,
        "alerts": int(pred.sum()),
    }


@st.cache_data(show_spinner=False)
def compute_cost_curve(prob, label, amount, cost_fp, grid=None):
    if grid is None:
        grid = np.round(np.linspace(0.01, 0.95, 95), 3)
    rows = [cost_at_threshold(prob, label, amount, t, cost_fp) for t in grid]
    return pd.DataFrame(rows)


def fmt_money(x, currency="USD", rate=1.0):
    if pd.isna(x):
        return "—"
    val = x * (rate if currency == "VND" else 1)
    if currency == "VND":
        return f"{val:,.0f} ₫"
    return f"${val:,.2f}"


def kpi_card(label, value, sub=None, kind=""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""<div class="kpi-card {kind}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {sub_html}
            </div>""",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — NGUỒN DỮ LIỆU, MÔ HÌNH, ĐIỀU HƯỚNG
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud DSS")
    st.caption("Hệ thống hỗ trợ ra quyết định phát hiện gian lận giao dịch thẻ tín dụng")
    st.markdown("---")

    st.markdown("**1. Mô hình dự đoán**")
    model_file = st.file_uploader("Tải mô hình (.pkl)", type=["pkl"], label_visibility="collapsed")
    if model_file is not None:
        bundle = load_model_bundle(model_file.getvalue())
        st.success("Đã tải mô hình tùy chỉnh.")
    elif DEFAULT_MODEL_PATH.exists():
        bundle = load_model_bundle(str(DEFAULT_MODEL_PATH))
        st.caption(f"Đang dùng mô hình mặc định: `{DEFAULT_MODEL_PATH.name}`")
    else:
        bundle = None
        st.warning("Chưa có mô hình. Vui lòng tải file .pkl.")

    st.markdown("**2. Dữ liệu giao dịch**")
    data_file = st.file_uploader("Tải dữ liệu (.csv)", type=["csv"], label_visibility="collapsed")
    use_demo = st.button("📊 Dùng dữ liệu mẫu để thử nghiệm", use_container_width=True)

    st.markdown("---")
    st.markdown("**3. Đơn vị tiền tệ hiển thị**")
    currency = st.radio("Đơn vị", ["USD", "VND"], horizontal=True, label_visibility="collapsed")
    fx_rate = 1.0
    if currency == "VND":
        fx_rate = st.number_input("Tỷ giá USD→VND", min_value=1000, max_value=50000, value=25400, step=100)

    st.markdown("---")
    page = st.radio(
        "ĐIỀU HƯỚNG",
        [
            "1️⃣ Tổng quan điều hành",
            "2️⃣ Tình hình giao dịch",
            "3️⃣ Nguyên nhân gian lận",
            "4️⃣ Danh sách cần xử lý",
            "5️⃣ Quyết định chính sách",
        ],
    )

# ──────────────────────────────────────────────────────────────────────────
# NẠP DỮ LIỆU
# ──────────────────────────────────────────────────────────────────────────
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if use_demo:
    st.session_state.demo_mode = True

raw_df = None
if data_file is not None:
    raw_df = pd.read_csv(data_file)
    st.session_state.demo_mode = False
elif st.session_state.demo_mode:
    raw_df = generate_demo_data()

if bundle is None:
    st.title("🛡️ Hệ thống hỗ trợ ra quyết định phát hiện gian lận")
    st.info("👈 Vui lòng tải mô hình `.pkl` ở thanh bên để bắt đầu.")
    st.stop()

if raw_df is None:
    st.title("🛡️ Hệ thống hỗ trợ ra quyết định phát hiện gian lận")
    st.info("👈 Vui lòng tải lên dữ liệu giao dịch (.csv) hoặc bấm **“Dùng dữ liệu mẫu”** ở thanh bên để bắt đầu.")
    with st.expander("ℹ️ Cấu trúc dữ liệu khuyến nghị"):
        st.markdown(
            "**Tải lên file CSV thô từ Kaggle là đủ** — hệ thống tự tính tất cả đặc trưng:\n\n"
            "| Cột gốc (raw) | Đặc trưng được tính tự động |\n"
            "|---|---|\n"
            "| `trans_date_trans_time` | `trans_hour` → `sin_hour`, `cos_hour`, `is_night` |\n"
            "| `dob` | `age` (năm GD − năm sinh) |\n"
            "| `amt` | `amt_log`, `amt_ratio`, `amt_log_x_is_night` |\n"
            "| `lat`, `long`, `merch_lat`, `merch_long` | `distance_km` (Haversine) |\n"
            "| `category` | `is_online` (chứa 'net'), `category_risk_tier` |\n"
            "| `merchant` | `merchant_freq` (tần suất xuất hiện) |\n"
            "| `cc_num` + `unix_time` | `customer_avg_spending`, `customer_total_spending`, `time_since_last_txn`, `txn_count_24h` |\n\n"
            "Cột `is_fraud` *(tùy chọn)* — cần để phân tích chi phí & tối ưu ngưỡng cảnh báo."
        )
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]
categorical_features = bundle.get("categorical_features", [])
pandas_categorical = get_pandas_categorical(model)
model_best_threshold = float(bundle.get("best_threshold", 0.16))
model_cost_fp_default = 5.0
try:
    model_cost_fp_default = float(bundle.get("eval_metrics", {}).get("estimated_cost", 9360) and 5.0)
except Exception:
    pass

X, df_full, notes = prepare_features(raw_df, feature_cols, categorical_features, pandas_categorical)
proba = model.predict_proba(X)[:, 1]
df_full["fraud_probability"] = proba

amount_col = find_col(df_full, CANDIDATE_AMOUNT_COLS)
if amount_col is None:
    df_full["amount"] = np.expm1(df_full["amt_log"]) if "amt_log" in df_full.columns else 1.0
    amount_col = "amount"
    notes.append("Không tìm thấy cột giá trị giao dịch — ước tính `amount` từ `amt_log` (chỉ mang tính tham khảo).")

label_col = find_col(df_full, CANDIDATE_LABEL_COLS)
has_ground_truth = label_col is not None
merchant_col = find_col(df_full, CANDIDATE_MERCHANT_COLS)
country_col = find_col(df_full, CANDIDATE_COUNTRY_COLS)
hour_series = reconstruct_hour(df_full)
df_full["_hour"] = hour_series

# ── Trạng thái ngưỡng & chi phí dùng chung toàn hệ thống ──
if "threshold" not in st.session_state:
    st.session_state.threshold = model_best_threshold
if "cost_fp" not in st.session_state:
    st.session_state.cost_fp = model_cost_fp_default

risk_label, decision = assign_decision(df_full["fraud_probability"].values, st.session_state.threshold)
df_full["risk_level"] = risk_label
df_full["decision"] = decision

# Đường cong chi phí & ngưỡng tối ưu (chỉ khi có nhãn thực tế)
cost_curve = None
optimal_row = None
current_row = None
if has_ground_truth:
    label_arr = df_full[label_col].astype(int).values
    amount_arr = df_full[amount_col].astype(float).values
    cost_curve = compute_cost_curve(df_full["fraud_probability"].values, label_arr, amount_arr, st.session_state.cost_fp)
    optimal_row = cost_curve.loc[cost_curve["total_cost"].idxmin()]
    current_row = pd.Series(
        cost_at_threshold(df_full["fraud_probability"].values, label_arr, amount_arr, st.session_state.threshold, st.session_state.cost_fp)
    )

if notes:
    with st.expander(f"⚠️ Ghi chú tiền xử lý dữ liệu ({len(notes)})", expanded=False):
        for n in notes:
            st.markdown(f"- {n}")

# ──────────────────────────────────────────────────────────────────────────
# TRANG 1 — TỔNG QUAN ĐIỀU HÀNH (EXECUTIVE SUMMARY)
# ──────────────────────────────────────────────────────────────────────────
def page_executive_summary():
    st.markdown('<div class="section-eyebrow">TRANG 1 · DÀNH CHO NHÀ QUẢN LÝ</div>', unsafe_allow_html=True)
    st.title("Tổng quan điều hành")

    total_txn = len(df_full)
    alerts = int((df_full["risk_level"].isin(["High", "Very High"])).sum())
    alert_rate = alerts / total_txn * 100 if total_txn else 0

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Tổng giao dịch", f"{total_txn:,}", "Đã được phân tích")
    with cols[1]:
        kpi_card("Giao dịch nguy cơ cao", f"{alerts:,}", "Mức High & Very High", kind="alert")
    with cols[2]:
        kpi_card("Tỷ lệ cảnh báo", f"{alert_rate:.2f}%", "trên tổng số giao dịch")
    with cols[3]:
        if has_ground_truth:
            kpi_card(
                "Chi phí ước tính hiện tại",
                fmt_money(current_row["total_cost"], currency, fx_rate),
                f"Tại ngưỡng {st.session_state.threshold:.2f}",
                kind="good",
            )
        else:
            kpi_card("Chi phí ước tính", "Cần dữ liệu nhãn", "Tải cột `is_fraud` để tính chi phí")

    st.markdown("")

    if has_ground_truth:
        saving = current_row["total_cost"] - optimal_row["total_cost"]
        saving_pct = (saving / current_row["total_cost"] * 100) if current_row["total_cost"] > 0 else 0
        if saving > max(1.0, current_row["total_cost"] * 0.01):
            st.markdown(
                f"""<div class="reco-banner">
                    <h3>💡 Khuyến nghị hành động</h3>
                    <p>Hệ thống đề xuất chuyển ngưỡng cảnh báo từ <b>{st.session_state.threshold:.2f}</b>
                    sang <b>{optimal_row['threshold']:.2f}</b>, giúp giảm chi phí xử lý dự kiến từ
                    <b>{fmt_money(current_row['total_cost'], currency, fx_rate)}</b> xuống
                    <b>{fmt_money(optimal_row['total_cost'], currency, fx_rate)}</b>
                    (tiết kiệm khoảng <b>{saving_pct:.0f}%</b>).
                    Xem chi tiết tại trang “Quyết định chính sách”.</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="reco-banner">
                    <h3>✅ Chính sách hiện tại đã gần tối ưu</h3>
                    <p>Ngưỡng cảnh báo hiện tại ({st.session_state.threshold:.2f}) đang mang lại chi phí gần với
                    mức thấp nhất có thể đạt được ({fmt_money(optimal_row['total_cost'], currency, fx_rate)}).
                    Chưa cần thay đổi chính sách.</p>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """<div class="note-box">ℹ️ Dữ liệu hiện tại không có cột nhãn thực tế (ví dụ <code>is_fraud</code>),
            nên hệ thống chưa thể tính chi phí thực tế hoặc đề xuất ngưỡng tối ưu một cách định lượng.
            Bạn vẫn có thể xem danh sách giao dịch theo mức rủi ro ở trang 4.</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("###")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Cơ cấu mức độ rủi ro của các giao dịch**")
        risk_counts = df_full["risk_level"].value_counts().reindex(["Low", "Medium", "High", "Very High"]).fillna(0)
        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map=RISK_COLOR,
            labels={"x": "Mức rủi ro", "y": "Số giao dịch"},
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Quyết định đề xuất**")
        dec_counts = df_full["decision"].value_counts()
        fig2 = px.pie(values=dec_counts.values, names=dec_counts.index, hole=0.55)
        fig2.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# TRANG 2 — TÌNH HÌNH GIAO DỊCH (DESCRIPTIVE ANALYTICS)
# ──────────────────────────────────────────────────────────────────────────
def page_descriptive():
    st.markdown('<div class="section-eyebrow">TRANG 2 · ĐIỀU GÌ ĐANG XẢY RA?</div>', unsafe_allow_html=True)
    st.title("Tình hình giao dịch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Phân bố giao dịch theo giờ**")
        hour_df = df_full.dropna(subset=["_hour"]).copy()
        hour_df["_hour_int"] = hour_df["_hour"].round().astype(int) % 24
        h_counts = hour_df["_hour_int"].value_counts().sort_index()
        fig = px.bar(x=h_counts.index, y=h_counts.values, labels={"x": "Giờ trong ngày", "y": "Số giao dịch"})
        fig.update_traces(marker_color=PRIMARY)
        fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Phân bố giá trị giao dịch**")
        fig = px.histogram(df_full, x=amount_col, nbins=40, labels={amount_col: "Giá trị giao dịch"})
        fig.update_traces(marker_color=ACCENT_GOOD)
        fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if merchant_col:
            st.markdown("**Phân bố theo nhóm Merchant**")
            m_counts = df_full[merchant_col].value_counts().head(10)
            fig = px.bar(x=m_counts.values, y=m_counts.index, orientation="h")
            fig.update_traces(marker_color=PRIMARY)
            fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="", xaxis_title="Số giao dịch")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có cột Merchant trong dữ liệu để thống kê.")
    with c4:
        if country_col:
            st.markdown("**Phân bố theo khu vực / quốc gia**")
            c_counts = df_full[country_col].value_counts().head(10)
            fig = px.bar(x=c_counts.values, y=c_counts.index, orientation="h")
            fig.update_traces(marker_color=ACCENT)
            fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="", xaxis_title="Số giao dịch")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có cột khu vực/quốc gia trong dữ liệu để thống kê.")

    st.markdown("**Phân bố theo loại giao dịch (Online / Offline)**")
    if "is_online" in df_full.columns:
        online_counts = df_full["is_online"].map({1: "Trực tuyến", 0: "Tại điểm bán"}).value_counts()
        fig = px.bar(x=online_counts.index, y=online_counts.values, color=online_counts.index)
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# TRANG 3 — NGUYÊN NHÂN GIAN LẬN (DIAGNOSTIC ANALYTICS)
# ──────────────────────────────────────────────────────────────────────────
def page_diagnostic():
    st.markdown('<div class="section-eyebrow">TRANG 3 · TẠI SAO GIAN LẬN XẢY RA?</div>', unsafe_allow_html=True)
    st.title("Phân tích nguyên nhân")

    if has_ground_truth:
        fraud_df = df_full[df_full[label_col].astype(int) == 1].copy()
        basis_note = "dựa trên giao dịch **gian lận thực tế** (`is_fraud = 1`)"
    else:
        fraud_df = df_full[df_full["risk_level"].isin(["High", "Very High"])].copy()
        basis_note = "dựa trên giao dịch **được mô hình dự đoán rủi ro cao** (chưa có nhãn thực tế để xác nhận)"

    st.caption(f"Các thống kê dưới đây {basis_note}.")

    if len(fraud_df) == 0:
        st.warning("Không có giao dịch nào để phân tích nguyên nhân (0 giao dịch gian lận/nguy cơ cao).")
        return  # ← dừng sớm, tránh lỗi ở các bước bên dưới khi fraud_df rỗng

    total_fraud = len(fraud_df)
    night_pct = fraud_df["is_night"].mean() * 100 if "is_night" in fraud_df.columns else np.nan
    online_pct = fraud_df["is_online"].mean() * 100 if "is_online" in fraud_df.columns else np.nan
    intl_pct = (
        (fraud_df[country_col] != fraud_df[country_col].mode()[0]).mean() * 100
        if country_col and country_col in fraud_df.columns
        else np.nan
    )
    high_tier_pct = (
        (fraud_df["category_risk_tier"] == "High").mean() * 100
        if "category_risk_tier" in fraud_df.columns
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Xảy ra vào ban đêm", f"{night_pct:.0f}%" if not np.isnan(night_pct) else "—", "22h–6h", kind="alert")
    with c2:
        kpi_card("Giao dịch trực tuyến", f"{online_pct:.0f}%" if not np.isnan(online_pct) else "—")
    with c3:
        kpi_card("Nhóm rủi ro Merchant cao", f"{high_tier_pct:.0f}%" if not np.isnan(high_tier_pct) else "—", kind="alert")
    with c4:
        kpi_card("Giao dịch ngoài khu vực phổ biến", f"{intl_pct:.0f}%" if not np.isnan(intl_pct) else "—")

    st.markdown("###")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Theo khung giờ**")
        # Kiểm tra cột _hour tồn tại và không toàn NaN trước khi dùng
        if "_hour" in fraud_df.columns:
            hd = fraud_df.dropna(subset=["_hour"]).copy()
        else:
            hd = pd.DataFrame()
        if len(hd) > 0:
            hd["_hour_int"] = hd["_hour"].round().astype(int) % 24
            hc = hd["_hour_int"].value_counts().sort_index()
            # px.area trả về Scatter với fill='tozeroy' — fillcolor phải là chuỗi rgba, KHÔNG phải hex
            # (Plotly validator từ chối hex string cho fillcolor của Scatter)
            r, g, b = int(ACCENT[1:3], 16), int(ACCENT[3:5], 16), int(ACCENT[5:7], 16)
            fig = px.area(x=hc.index, y=hc.values, labels={"x": "Giờ", "y": "Số giao dịch"})
            fig.update_traces(
                line=dict(color=ACCENT),
                fillcolor=f"rgba({r},{g},{b},0.15)"
            )
            fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu giờ để hiển thị.")
    with c2:
        st.markdown("**Theo nhóm rủi ro Merchant (category_risk_tier)**")
        if "category_risk_tier" in fraud_df.columns:
            tier_c = fraud_df["category_risk_tier"].value_counts()
            fig = px.pie(values=tier_c.values, names=tier_c.index, hole=0.5)
            fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # TỐI ƯU THRESHOLD THEO CHI PHÍ THỰC — theo đúng công thức Pipeline 2:
    #   Tổng chi phí = Σ amt[FN] + cost_fp × số_FP
    #   → tìm threshold tối thiểu hoá tổng chi phí này
    # ──────────────────────────────────────────────────────────────────
    if has_ground_truth:
        st.markdown("###")
        st.markdown("**Tối ưu ngưỡng cảnh báo theo chi phí thực tế (Cost-Sensitive Threshold)**")
        st.caption(
            "Công thức: **Tổng chi phí = Σ số tiền bị bỏ sót (FN) + chi phí xử lý nhầm × số FP** "
            "— nhất quán với Pipeline 2 (LightGBM Cost-Sensitive)."
        )

        label_arr_diag  = df_full[label_col].astype(int).values
        amount_arr_diag = df_full[amount_col].astype(float).values
        prob_arr_diag   = df_full["fraud_probability"].values

        # ── Tại sao cần auto-scale cost_fp? ──────────────────────────────────
        # Pipeline 2 dùng COST_PER_FALSE_ALARM = 5 USD trên tập ~1.3M dòng với
        # hàng trăm nghìn FP có thể xảy ra → FP chiếm phần đáng kể tổng chi
        # phí → đường cong có hình chữ U rõ ràng.
        # Trên tập inference nhỏ hơn, số FP ít hơn nhiều, nên nếu giữ nguyên
        # 5 USD tuyệt đối thì FN (mỗi cái = vài trăm USD) luôn thắng áp đảo
        # → đường cong chỉ tăng đơn điệu, không có điểm uốn.
        # Giải pháp: biểu diễn cost_fp theo % median amt fraud — scale tự động
        # theo phân phối của tập dữ liệu hiện tại, giữ nguyên ý nghĩa kinh tế.
        fraud_idxs = label_arr_diag == 1
        fraud_amt_median = float(np.median(amount_arr_diag[fraud_idxs])) if fraud_idxs.sum() > 0 else 100.0
        fraud_amt_median = max(fraud_amt_median, 1.0)

        cost_fp_pct = st.slider(
            "Chi phí xử lý 1 FP (% so với trung vị giá trị fraud)",
            min_value=1, max_value=200, value=10, step=1,
            key="diag_cost_fp",
            help=(
                f"Trung vị amt fraud trong tập này ≈ ${fraud_amt_median:,.0f}. "
                f"Ví dụ: 10% → ${fraud_amt_median*0.10:,.0f}/FP. "
                f"Pipeline 2 dùng $5/FP = tương đương {5/fraud_amt_median*100:.1f}% tập train."
            )
        )
        cost_fp_diag = fraud_amt_median * cost_fp_pct / 100.0
        st.caption(
            f"→ Chi phí FP hiệu dụng: **${cost_fp_diag:,.2f}** / cảnh báo sai "
            f"(trung vị amt fraud = ${fraud_amt_median:,.0f}, tỷ lệ {cost_fp_pct}%)"
        )

        # Tính đường cong chi phí trên lưới 99 điểm — nhất quán với Pipeline 2
        # Công thức: Tổng_chi_phí(t) = Σ amt[FN] + cost_fp × #FP
        # Với cost_fp được scale hợp lý → đường cong luôn có hình chữ U:
        #   threshold thấp  → nhiều FP → chi phí FP lớn
        #   threshold cao   → nhiều FN → chi phí FN lớn (tiền bị bỏ sót)
        #   điểm giữa tối ưu → cân bằng hai loại chi phí
        thresholds_grid = np.round(np.linspace(0.01, 0.99, 99), 3)
        total_costs = []
        for t in thresholds_grid:
            pred_t  = (prob_arr_diag >= t).astype(int)
            fn_mask = (label_arr_diag == 1) & (pred_t == 0)
            fp_mask = (label_arr_diag == 0) & (pred_t == 1)
            total_costs.append(amount_arr_diag[fn_mask].sum() + cost_fp_diag * fp_mask.sum())
        total_costs = np.array(total_costs)

        best_idx        = int(np.argmin(total_costs))
        best_threshold  = thresholds_grid[best_idx]
        best_cost       = total_costs[best_idx]
        default_idx     = int(np.argmin(np.abs(thresholds_grid - 0.5)))
        default_cost    = total_costs[default_idx]
        current_idx     = int(np.argmin(np.abs(thresholds_grid - st.session_state.threshold)))
        current_cost    = total_costs[current_idx]
        saving          = current_cost - best_cost
        saving_pct      = (saving / current_cost * 100) if current_cost > 0 else 0

        # Biểu đồ đường cong chi phí
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=thresholds_grid, y=total_costs,
            mode="lines", line=dict(color=PRIMARY, width=3), name="Chi phí ước tính"
        ))
        fig_cost.add_vline(
            x=best_threshold, line_dash="dot", line_color=ACCENT_GOOD,
            annotation_text=f"Tối ưu: {best_threshold:.2f} (${best_cost:,.0f})",
            annotation_font_color=ACCENT_GOOD,
        )
        fig_cost.add_vline(
            x=st.session_state.threshold, line_dash="dash", line_color=ACCENT,
            annotation_text=f"Hiện tại: {st.session_state.threshold:.2f} (${current_cost:,.0f})",
            annotation_font_color=ACCENT,
        )
        fig_cost.update_layout(
            height=360,
            margin=dict(t=30, b=10, l=10, r=10),
            xaxis_title="Ngưỡng phân loại (Threshold)",
            yaxis_title="Tổng chi phí ước tính (USD)",
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # So sánh KPI tại threshold hiện tại vs tối ưu
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            kpi_card("Threshold mặc định (0.5)", fmt_money(default_cost, currency, fx_rate), "Chi phí ước tính")
        with kc2:
            kpi_card("Threshold hiện tại", fmt_money(current_cost, currency, fx_rate),
                     f"t = {st.session_state.threshold:.2f}")
        with kc3:
            kpi_card("Threshold tối ưu", fmt_money(best_cost, currency, fx_rate),
                     f"t = {best_threshold:.2f}", kind="good")
        with kc4:
            kpi_card("Tiết kiệm ước tính", fmt_money(saving, currency, fx_rate),
                     f"{saving_pct:.1f}% so với ngưỡng hiện tại", kind="good")

        if saving > max(1.0, current_cost * 0.01):
            st.markdown(
                f"""<div class="reco-banner">
                    <h3>💡 Khuyến nghị từ phân tích nguyên nhân</h3>
                    <p>Dùng ngưỡng <b>{best_threshold:.2f}</b> thay cho <b>{st.session_state.threshold:.2f}</b>
                    hiện tại — ước tính tiết kiệm <b>{fmt_money(saving, currency, fx_rate)}</b>
                    ({saving_pct:.0f}% chi phí xử lý). Áp dụng tại trang "Quyết định chính sách".</p>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"✅ Áp dụng ngưỡng tối ưu ({best_threshold:.2f}) ngay", key="diag_apply_threshold"):
                st.session_state.threshold = float(best_threshold)
                st.rerun()
        else:
            st.success(f"Ngưỡng hiện tại ({st.session_state.threshold:.2f}) đã gần mức tối ưu — chưa cần thay đổi.")
    else:
        st.markdown("###")
        st.markdown(
            """<div class="note-box">ℹ️ Tải thêm cột nhãn thực tế (<code>is_fraud</code>)
            để xem phân tích tối ưu ngưỡng theo chi phí tại trang này.</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("###")
    st.markdown("**Các đặc trưng ảnh hưởng mạnh nhất tới quyết định của mô hình**")
    st.caption("Information Gain — mức độ đặc trưng giúp mô hình phân biệt giao dịch gian lận / hợp lệ.")
    fi = bundle.get("feature_importance")
    if fi:
        fi_df = pd.DataFrame(fi).sort_values("gain", ascending=True).tail(15)
        fig = px.bar(fi_df, x="gain", y="feature", orientation="h")
        fig.update_traces(marker_color=ACCENT_GOOD)
        fig.update_layout(height=440, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="", xaxis_title="Gain")
        st.plotly_chart(fig, use_container_width=True)

    if merchant_col and len(fraud_df):
        st.markdown("**Merchant có nhiều giao dịch rủi ro nhất**")
        mc = fraud_df[merchant_col].value_counts().head(8)
        fig = px.bar(x=mc.values, y=mc.index, orientation="h")
        fig.update_traces(marker_color=PRIMARY)
        fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="", xaxis_title="Số giao dịch")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# TRANG 4 — DANH SÁCH CẦN XỬ LÝ (PREDICTIVE ANALYTICS)
# ──────────────────────────────────────────────────────────────────────────
def page_predictive():
    st.markdown('<div class="section-eyebrow">TRANG 4 · NHỮNG GIAO DỊCH NÀO CẦN ĐƯỢC KIỂM TRA?</div>', unsafe_allow_html=True)
    st.title("Danh sách giao dịch cần xử lý")
    st.caption(f"Đang áp dụng ngưỡng cảnh báo hiện tại: **{st.session_state.threshold:.2f}** (điều chỉnh tại trang 5).")

    levels = st.multiselect(
        "Lọc theo mức rủi ro", ["Very High", "High", "Medium", "Low"], default=["Very High", "High"]
    )
    search = st.text_input("🔎 Tìm theo mã giao dịch / merchant (tùy chọn)")

    view = df_full.copy()
    if levels:
        view = view[view["risk_level"].isin(levels)]
    if search:
        mask = pd.Series(False, index=view.index)
        id_col = find_col(view, CANDIDATE_ID_COLS)
        if id_col:
            mask |= view[id_col].astype(str).str.contains(search, case=False, na=False)
        if merchant_col:
            mask |= view[merchant_col].astype(str).str.contains(search, case=False, na=False)
        view = view[mask]

    id_col = find_col(view, CANDIDATE_ID_COLS) or view.index.name
    display_cols = []
    if id_col and id_col in view.columns:
        display_cols.append(id_col)
    display_cols += [c for c in [amount_col, merchant_col, country_col] if c and c not in display_cols]
    display_cols += ["fraud_probability", "risk_level", "decision"]
    if has_ground_truth:
        display_cols.append(label_col)

    show_df = view[display_cols].sort_values("fraud_probability", ascending=False).reset_index(drop=True)
    show_df = show_df.rename(
        columns={
            amount_col: "Giá trị GD",
            merchant_col: "Merchant" if merchant_col else merchant_col,
            country_col: "Khu vực" if country_col else country_col,
            "fraud_probability": "Xác suất gian lận",
            "risk_level": "Mức rủi ro",
            "decision": "Quyết định đề xuất",
            label_col: "Thực tế (is_fraud)" if has_ground_truth else label_col,
        }
    )

    st.caption(f"Đang hiển thị **{len(show_df):,}** / {len(df_full):,} giao dịch.")
    st.dataframe(
        show_df.style.format({"Xác suất gian lận": "{:.1%}", "Giá trị GD": "{:,.2f}"} if "Giá trị GD" in show_df.columns else {"Xác suất gian lận": "{:.1%}"}),
        use_container_width=True,
        height=480,
    )

    csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Tải danh sách này (CSV)", data=csv_bytes, file_name="giao_dich_can_xu_ly.csv", mime="text/csv"
    )


# ──────────────────────────────────────────────────────────────────────────
# TRANG 5 — QUYẾT ĐỊNH CHÍNH SÁCH (PRESCRIPTIVE ANALYTICS)
# ──────────────────────────────────────────────────────────────────────────
def page_prescriptive():
    st.markdown('<div class="section-eyebrow">TRANG 5 · NÊN LÀM GÌ?</div>', unsafe_allow_html=True)
    st.title("Quyết định chính sách cảnh báo")
    st.caption("Cân bằng giữa chi phí kiểm tra nhầm (False Positive) và rủi ro bỏ sót gian lận (False Negative).")

    if not has_ground_truth:
        st.markdown(
            f"""<div class="note-box">⚠️ Dữ liệu hiện tại không có cột nhãn thực tế
            (thử các tên cột: {', '.join(f'`{c}`' for c in CANDIDATE_LABEL_COLS)}). Không thể tính chi phí thực tế
            hoặc đề xuất ngưỡng tối ưu một cách định lượng. Bạn vẫn có thể điều chỉnh ngưỡng để xem số lượng
            cảnh báo phát sinh.</div>""",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        new_threshold = st.slider(
            "Ngưỡng phân loại cảnh báo (Threshold)", 0.01, 0.95, float(st.session_state.threshold), 0.01
        )
    with c2:
        new_cost_fp = st.slider(
            "Chi phí xử lý 1 cảnh báo sai (USD)", 0.5, 50.0, float(st.session_state.cost_fp), 0.5
        )

    if new_threshold != st.session_state.threshold or new_cost_fp != st.session_state.cost_fp:
        st.session_state.threshold = new_threshold
        st.session_state.cost_fp = new_cost_fp
        st.rerun()

    alerts_now = int((df_full["fraud_probability"] >= st.session_state.threshold).sum())

    if has_ground_truth:
        label_arr = df_full[label_col].astype(int).values
        amount_arr = df_full[amount_col].astype(float).values
        curr = cost_at_threshold(df_full["fraud_probability"].values, label_arr, amount_arr, st.session_state.threshold, st.session_state.cost_fp)
        curve = compute_cost_curve(df_full["fraud_probability"].values, label_arr, amount_arr, st.session_state.cost_fp)
        opt = curve.loc[curve["total_cost"].idxmin()]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Số cảnh báo", f"{curr['alerts']:,}")
        with c2:
            kpi_card("Cảnh báo sai (FP)", f"{curr['fp_count']:,}", kind="alert")
        with c3:
            kpi_card("Gian lận bị bỏ sót (FN)", f"{curr['fn_count']:,}", kind="alert")
        with c4:
            kpi_card("Tổng chi phí dự kiến", fmt_money(curr["total_cost"], currency, fx_rate), kind="good")

        st.markdown("###")
        st.markdown("**Chi phí dự kiến theo từng mức ngưỡng**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["total_cost"], mode="lines", line=dict(color=PRIMARY, width=3), name="Chi phí"))
        fig.add_vline(x=st.session_state.threshold, line_dash="dash", line_color=ACCENT, annotation_text="Hiện tại")
        fig.add_vline(x=opt["threshold"], line_dash="dot", line_color=ACCENT_GOOD, annotation_text="Tối ưu")
        fig.update_layout(
            height=380, margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Ngưỡng phân loại", yaxis_title=f"Chi phí ước tính ({currency})",
        )
        st.plotly_chart(fig, use_container_width=True)

        saving = curr["total_cost"] - opt["total_cost"]
        saving_pct = (saving / curr["total_cost"] * 100) if curr["total_cost"] > 0 else 0

        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"""
**Ngưỡng hiện tại: `{st.session_state.threshold:.2f}`**
- Chi phí: {fmt_money(curr['total_cost'], currency, fx_rate)}
- Precision: {curr['precision']:.1%} · Recall: {curr['recall']:.1%}
""")
        with colB:
            st.markdown(f"""
**Ngưỡng tối ưu: `{opt['threshold']:.2f}`**
- Chi phí: {fmt_money(opt['total_cost'], currency, fx_rate)}
- Precision: {opt['precision']:.1%} · Recall: {opt['recall']:.1%}
""")

        if saving > max(1.0, curr["total_cost"] * 0.01):
            st.markdown(
                f"""<div class="reco-banner">
                    <h3>💡 Khuyến nghị</h3>
                    <p>Áp dụng ngưỡng <b>{opt['threshold']:.2f}</b> thay cho <b>{st.session_state.threshold:.2f}</b>
                    hiện tại — ước tính giảm <b>{saving_pct:.0f}%</b> chi phí xử lý, tương ứng tiết kiệm
                    <b>{fmt_money(saving, currency, fx_rate)}</b>.</p>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"✅ Áp dụng ngưỡng tối ưu ({opt['threshold']:.2f})"):
                st.session_state.threshold = float(opt["threshold"])
                st.rerun()
        else:
            st.success("Ngưỡng hiện tại đã gần mức tối ưu — chưa cần thay đổi chính sách.")
    else:
        kpi_card("Số cảnh báo phát sinh", f"{alerts_now:,}", f"trên tổng {len(df_full):,} giao dịch tại ngưỡng {st.session_state.threshold:.2f}")
        st.info("Tải thêm cột nhãn thực tế (`is_fraud`) để xem phân tích chi phí và đề xuất ngưỡng tối ưu đầy đủ.")


# ──────────────────────────────────────────────────────────────────────────
# ĐIỀU HƯỚNG
# ──────────────────────────────────────────────────────────────────────────
PAGES = {
    "1️⃣ Tổng quan điều hành": page_executive_summary,
    "2️⃣ Tình hình giao dịch": page_descriptive,
    "3️⃣ Nguyên nhân gian lận": page_diagnostic,
    "4️⃣ Danh sách cần xử lý": page_predictive,
    "5️⃣ Quyết định chính sách": page_prescriptive,
}
PAGES[page]()

st.markdown("---")
st.caption(
    "Hệ thống hỗ trợ ra quyết định (DSS) phát hiện gian lận giao dịch thẻ tín dụng · "
    "Mô hình: LightGBM + Cost-Sensitive Learning"
)
