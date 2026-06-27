import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# CẤU HÌNH
# ================================================================
st.set_page_config(
    page_title="Fraud Detection DSS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BEST_THRESHOLD = 0.16  # ngưỡng tối ưu 

CATEGORY_RISK = {
    "shopping_net": 2, "misc_net": 2, "grocery_pos": 2,
    "shopping_pos": 1, "gas_transport": 1, "misc_pos": 1,
    "travel": 1,       "entertainment": 1,
    "food_dining": 0,  "health_fitness": 0, "personal_care": 0,
    "home": 0,         "kids_pets": 0,      "grocery_net": 0,
}

# ================================================================
# CSS — light, tối giản
# ================================================================
st.markdown("""
<style>
/* nền */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main { background:#f7f8fa !important; color:#111827 !important; }
[data-testid="stHeader"]  { background:#f7f8fa !important; }
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background:#ffffff !important;
    border-right:1px solid #e5e7eb !important;
}
[data-testid="stSidebar"] * { color:#111827 !important; }
.block-container { padding-top:1.4rem !important; }

/* file uploader */
[data-testid="stFileUploaderDropzone"] {
    background:#ffffff !important;
    border:1.5px dashed #d1d5db !important;
    border-radius:8px !important;
}
[data-testid="stFileUploaderDropzone"] * { color:#374151 !important; }
[data-testid="stFileUploaderDropzone"] button {
    background:#2563eb !important; color:#fff !important; border:none !important;
}

/* inputs */
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] button { background:#fff !important; color:#111827 !important; }
[data-baseweb="select"] div          { background:#fff !important; color:#111827 !important; }
[data-baseweb="input"] input         { background:#fff !important; color:#111827 !important; }

/* expander */
[data-testid="stExpander"] {
    background:#fff !important; border:1px solid #e5e7eb !important; border-radius:8px !important;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:4px; background:transparent !important; }
.stTabs [data-baseweb="tab"] {
    background:#fff !important; border:1px solid #e5e7eb !important;
    border-radius:6px 6px 0 0; padding:6px 20px;
    color:#6b7280 !important; font-weight:500;
}
.stTabs [aria-selected="true"] {
    background:#2563eb !important; color:#fff !important; border-color:#2563eb !important;
}

/* kpi card */
.kpi {
    background:#fff; border:1px solid #e5e7eb; border-radius:10px;
    padding:16px 20px; text-align:center;
}
.kpi-val { font-size:1.75rem; font-weight:700; margin:4px 0; }
.kpi-lbl { font-size:.72rem; color:#6b7280 !important; text-transform:uppercase; letter-spacing:.06em; }
.kpi-sub { font-size:.78rem; color:#9ca3af !important; margin-top:3px; }

/* section header */
.sec {
    font-size:.92rem; font-weight:600; color:#111827;
    border-left:3px solid #2563eb; padding-left:9px; margin:18px 0 10px;
}

/* model info card (sidebar) */
.model-card {
    background:#f0f4ff; border:1px solid #c7d7fd; border-radius:8px; padding:12px 14px;
}
.model-card p { margin:3px 0; font-size:.83rem; color:#1e40af !important; }
.model-card b { color:#1e3a8a !important; }

/* risk alerts */
.a-hi  { background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; padding:12px 16px; }
.a-md  { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px; padding:12px 16px; }
.a-lo  { background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:12px 16px; }
.a-hi *,.a-hi b { color:#991b1b !important; }
.a-md *,.a-md b { color:#92400e !important; }
.a-lo *,.a-lo b { color:#166534 !important; }

/* chips */
.chip-r {
    display:inline-block; background:#fef2f2; border:1px solid #fca5a5;
    border-radius:20px; padding:3px 12px; font-size:.78rem; color:#991b1b !important; margin:3px;
}
.chip-s {
    display:inline-block; background:#f0fdf4; border:1px solid #86efac;
    border-radius:20px; padding:3px 12px; font-size:.78rem; color:#166534 !important; margin:3px;
}

/* table highlight classes (applied via pandas Styler) */
.stDataFrame { border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# LOAD MODEL
# ================================================================
@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_model():
    with open("fraud_lightgbm_dss.pkl", "rb") as f:
        return pickle.load(f)

payload      = load_model()
model        = payload["model"]
FEATURE_COLS = payload["feature_cols"]
CAT_COLS     = payload["categorical_features"]
METRICS      = payload["eval_metrics"]
FI_DF        = pd.DataFrame(payload["feature_importance"])


# ================================================================
# FEATURE ENGINEERING — khớp đúng pipeline training
# ================================================================
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # parse datetime
    for col in ["trans_date_trans_time", "dob"]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # amt_log
    df["amt_log"] = np.log1p(df["amt"].clip(lower=0)) if "amt" in df.columns else 0.0

    # khoảng cách địa lý (Haversine)
    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        def haversine(la1, lo1, la2, lo2):
            R = 6371.0
            la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
            return R * 2 * np.arcsin(np.sqrt(
                np.sin((la2 - la1) / 2) ** 2
                + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
            ))
        df["distance_km"] = haversine(
            df["lat"].fillna(0), df["long"].fillna(0),
            df["merch_lat"].fillna(0), df["merch_long"].fillna(0)
        )
    elif "distance_km" not in df.columns:
        df["distance_km"] = 75.0

    # thời gian
    if "trans_date_trans_time" in df.columns:
        h = df["trans_date_trans_time"].dt.hour.fillna(12).astype(float)
        df["sin_hour"] = np.sin(2 * np.pi * h / 24)
        df["cos_hour"] = np.cos(2 * np.pi * h / 24)
        df["is_night"] = ((h >= 22) | (h <= 3)).astype(int)
    else:
        df["sin_hour"] = df["cos_hour"] = df["is_night"] = 0

    # tuổi
    if "dob" in df.columns and "trans_date_trans_time" in df.columns:
        df["age"] = (
            (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
        ).fillna(40).clip(18, 100)
    elif "age" not in df.columns:
        df["age"] = 40.0

    # danh mục
    if "category" in df.columns:
        df["is_online"]          = df["category"].str.contains("net", na=False).astype(int)
        df["category_risk_tier"] = df["category"].map(CATEGORY_RISK).fillna(1).astype("category")
    else:
        df["is_online"] = 0
        df["category_risk_tier"] = pd.Categorical([1] * len(df))

    # gender → category
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype("category")

    # tần suất merchant
    if "merchant" in df.columns:
        df["merchant_freq"] = df["merchant"].map(
            df["merchant"].value_counts()
        ).fillna(1).astype(float)
    elif "merchant_freq" not in df.columns:
        df["merchant_freq"] = 1.0

    # hành vi — dùng giá trị file nếu có, không thì ước lượng
    if "customer_avg_spending" not in df.columns:
        df["customer_avg_spending"] = float(
            df["amt"].mean() if "amt" in df.columns else 70.0
        )
    if "amt_ratio" not in df.columns:
        df["amt_ratio"] = (
            (df["amt"] / df["customer_avg_spending"].clip(lower=1)).clip(0, 50)
            if "amt" in df.columns else 1.0
        )
    if "time_since_last_txn" not in df.columns:
        df["time_since_last_txn"] = 30000.0
    if "txn_count_24h" not in df.columns:
        df["txn_count_24h"] = 1.0

    # interaction features
    df["amt_log_x_is_night"] = df["amt_log"] * df["is_night"]
    df["is_night_x_online"]  = df["is_night"] * df["is_online"]

    return df


def predict(df_raw: pd.DataFrame, threshold: float):
    df_feat = engineer(df_raw)
    # đảm bảo đủ cột
    for c in FEATURE_COLS:
        if c not in df_feat.columns:
            df_feat[c] = 0
    X = df_feat[FEATURE_COLS].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    return proba, pred


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection DSS")
    st.caption("Hệ thống Hỗ trợ Ra quyết định Phát hiện Gian lận")
    st.divider()

    st.markdown("**Ngưỡng phân loại**")
    threshold = st.slider(
        "Risk Score threshold",
        min_value=0.05, max_value=0.95,
        value=BEST_THRESHOLD, step=0.01,
        help=f"Ngưỡng tối ưu từ đồ án: {BEST_THRESHOLD} "
             f"(tối thiểu hóa tổng chi phí, chi phí cảnh báo sai = $5/GD)"
    )
    st.caption(f"Ngưỡng tối ưu từ training: **{BEST_THRESHOLD}**")

    st.divider()
    st.markdown("**Thông số mô hình**")
    def _f(key, default=0): return float(METRICS.get(key, default))
    st.markdown(f"""
<div class="model-card">
<p>Thuật toán: <b>LightGBM GBDT</b></p>
<p>Số đặc trưng: <b>{len(FEATURE_COLS)}</b></p>
<p>Threshold tối ưu: <b>{BEST_THRESHOLD}</b></p>
<p>PR-AUC: <b>{_f("pr_auc"):.4f}</b></p>
<p>PR-AUC 5-fold CV: <b>{_f("cv_pr_auc_mean"):.4f} ± {_f("cv_pr_auc_std"):.4f}</b></p>
<p>ROC-AUC: <b>{_f("roc_auc"):.4f}</b></p>
<p>Recall (số lượng): <b>{_f("recall_count")*100:.2f}%</b></p>
<p>Recall (theo $): <b>{_f("dollar_recall"):.2f}%</b></p>
<p>Precision (theo $): <b>{_f("dollar_precision"):.2f}%</b></p>
<p>Chi phí ước tính: <b>${_f("estimated_cost"):,.0f}</b></p>
</div>
""", unsafe_allow_html=True)


# ================================================================
# TABS
# ================================================================
tab1, tab2 = st.tabs(["🚨 Phát hiện gian lận", "🔎 Chi tiết giao dịch"])


# ================================================================
# TAB 1 — PHÁT HIỆN THEO LÔ
# ================================================================
with tab1:
    st.markdown("## Phát hiện gian lận theo lô")

    with st.expander("📋 Format CSV đầu vào"):
        st.markdown("""
**Bắt buộc:** `amt` · `category` · `gender` · `lat` · `long` · `merch_lat` · `merch_long`

**Tùy chọn** (có thì kết quả tốt hơn):
`trans_date_trans_time` · `dob` · `merchant` · `customer_avg_spending` · `amt_ratio` · `time_since_last_txn` · `txn_count_24h`

**Giá trị `category` hợp lệ:**
`shopping_net` · `misc_net` · `grocery_pos` · `shopping_pos` · `gas_transport` · `misc_pos` · `grocery_net` · `travel` · `entertainment` · `personal_care` · `kids_pets` · `food_dining` · `home` · `health_fitness`
""")

    uploaded = st.file_uploader("Kéo thả hoặc chọn file CSV giao dịch", type=["csv"])

    if uploaded is None:
        st.info("Upload file CSV để bắt đầu. Hệ thống sẽ tự động tính đặc trưng và chấm điểm rủi ro.")
    else:
        df_raw = pd.read_csv(uploaded)
        st.caption(f"Đã tải: **{len(df_raw):,} giao dịch**")

        with st.spinner("Đang tính đặc trưng và chấm điểm rủi ro…"):
            proba, pred = predict(df_raw, threshold)

        # gắn kết quả vào dataframe
        df_out = df_raw.copy()
        df_out["risk_score"] = np.round(proba, 4)
        df_out["fraud_pred"] = pred
        # risk_level và fraud_pred tính theo threshold từ sidebar
        # (được cập nhật lại mỗi khi user kéo slider)
        df_out["fraud_pred"] = (proba >= threshold).astype(int)

        # lưu session
        st.session_state["df_result"] = df_out
        st.session_state["proba"]     = proba

        # ── tính risk_level động theo threshold hiện tại ──
        MID = 0.5
        df_out["fraud_pred"] = (proba >= threshold).astype(int)
        df_out["risk_level"] = pd.cut(
            proba,
            bins=[-0.001, threshold, MID, 1.001],
            labels=["🟢 An toàn", "🟡 Trung bình", "🔴 Rất cao"]
        )

        # ── KPI ──────────────────────────────────────
        n        = len(df_out)
        n_fraud  = int(df_out["fraud_pred"].sum())
        f_amt    = df_out.loc[df_out["fraud_pred"] == 1, "amt"].sum() \
                   if "amt" in df_out.columns else 0.0
        avg_sc   = float(proba.mean())

        st.markdown('<div class="sec">Tổng kết</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, (lbl, val, color, sub) in zip(
            [c1, c2, c3, c4],
            [
                ("Tổng giao dịch",     f"{n:,}",             "#2563eb", ""),
                ("Nghi ngờ gian lận",  f"{n_fraud:,}",       "#dc2626", f"{n_fraud/n*100:.1f}% tổng số"),
                ("Thiệt hại ước tính", f"${f_amt:,.0f}",     "#d97706", "tổng GD bị gắn cờ"),
                ("Avg. Risk Score",    f"{avg_sc:.3f}",       "#7c3aed", "toàn bộ tập dữ liệu"),
            ]
        ):
            col.markdown(f"""
            <div class="kpi">
                <div class="kpi-lbl">{lbl}</div>
                <div class="kpi-val" style="color:{color}">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        # ── Biểu đồ ──────────────────────────────────
        st.markdown('<div class="sec">Phân tích phân bố</div>', unsafe_allow_html=True)
        ca, cb = st.columns([3, 2])

        with ca:
            # Scatter: amt vs risk_score
            samp = df_out.sample(min(800, len(df_out)), random_state=1)
            color_map = samp["fraud_pred"].map({0: "Hợp lệ", 1: "Gian lận"})
            fig_sc = px.scatter(
                samp,
                x="amt" if "amt" in samp.columns else samp.index,
                y="risk_score",
                color=color_map,
                color_discrete_map={"Hợp lệ": "#93c5fd", "Gian lận": "#f87171"},
                opacity=0.65,
                labels={"amt": "Số tiền ($)", "risk_score": "Risk Score", "color": ""},
                title="Risk Score theo Số tiền giao dịch",
            )
            fig_sc.add_hline(
                y=threshold, line_dash="dash", line_color="#f59e0b",
                annotation_text=f"Threshold {threshold}",
                annotation_position="top right",
            )
            fig_sc.update_layout(
                template="plotly_white", height=310,
                margin=dict(t=40, b=30, l=10, r=10),
                legend=dict(title="", orientation="h", y=1.1),
                title_font_size=13,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with cb:
            # Pie: risk level
            rc = df_out["risk_level"].value_counts().reset_index()
            rc.columns = ["level", "count"]
            fig_pie = px.pie(
                rc, names="level", values="count", hole=0.45,
                color="level",
                color_discrete_map={
                    "🟢 An toàn":    "#86efac",
                    "🟡 Trung bình": "#fcd34d",
                    "🔴 Rất cao":    "#f87171",
                },
                title="Phân bố mức độ rủi ro",
            )
            fig_pie.update_layout(
                template="plotly_white", height=310,
                margin=dict(t=40, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.12),
                title_font_size=13,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Feature importance
        st.markdown('<div class="sec">Tầm quan trọng đặc trưng (Information Gain)</div>',
                    unsafe_allow_html=True)
        fi_plot = FI_DF.sort_values("gain", ascending=True).tail(15)
        fig_fi = px.bar(
            fi_plot, x="gain", y="feature", orientation="h",
            color_discrete_sequence=["#2563eb"],
            labels={"gain": "Information Gain", "feature": ""},
        )
        fig_fi.update_layout(
            template="plotly_white", height=340,
            margin=dict(t=10, b=30, l=10, r=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── Bảng giao dịch ───────────────────────────
        st.markdown('<div class="sec">Danh sách giao dịch</div>', unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        only_fraud = f1.checkbox("Chỉ hiện giao dịch bị gắn cờ", value=True)
        cats = ["Tất cả"] + (
            sorted(df_out["category"].dropna().unique().tolist())
            if "category" in df_out.columns else []
        )
        sel_cat = f2.selectbox("Lọc danh mục", cats)

        df_show = df_out.copy()
        if only_fraud:
            df_show = df_show[df_show["fraud_pred"] == 1]
        if sel_cat != "Tất cả":
            df_show = df_show[df_show["category"] == sel_cat]

        disp_cols = [c for c in [
            "trans_date_trans_time", "category", "amt", "merchant",
            "risk_score", "risk_level", "fraud_pred"
        ] if c in df_show.columns]

        def highlight(row):
            s = row.get("risk_score", 0)
            if s >= 0.5:             return ["background-color:#fef2f2"] * len(row)
            if s >= BEST_THRESHOLD:  return ["background-color:#fffbeb"] * len(row)
            return [""] * len(row)

        fmt = {k: v for k, v in
               {"risk_score": "{:.4f}", "amt": "${:,.2f}"}.items()
               if k in disp_cols}

        st.caption(f"{len(df_show):,} giao dịch")
        st.dataframe(
            df_show[disp_cols].head(500).style.apply(highlight, axis=1).format(fmt),
            use_container_width=True, height=370,
        )

        st.download_button(
            "⬇️ Tải kết quả CSV",
            data=df_out.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )


# ================================================================
# TAB 2 — CHI TIẾT GIAO DỊCH
# ================================================================
with tab2:
    st.markdown("## Chi tiết & Thẩm định giao dịch")

    if "df_result" not in st.session_state:
        st.info("Vui lòng upload và chạy dự đoán ở tab **Phát hiện gian lận** trước.")
    else:
        df_res = st.session_state["df_result"]

        # danh sách giao dịch bị gắn cờ (ưu tiên) hoặc toàn bộ
        flagged = df_res[df_res["fraud_pred"] == 1]
        opts    = (flagged.index.tolist() if len(flagged) else df_res.index.tolist())[:300]

        cs1, cs2 = st.columns([4, 1])
        with cs1:
            sel = st.selectbox(
                f"Chọn giao dịch để thẩm định ({len(opts)} GD bị gắn cờ hiển thị)",
                options=opts,
                format_func=lambda i: (
                    f"#{i}  "
                    f"[{df_res.loc[i, 'category']}]  " if "category" in df_res.columns else f"#{i}  "
                ) + (
                    f"${df_res.loc[i, 'amt']:,.2f}  " if "amt" in df_res.columns else ""
                ) + f"— score {df_res.loc[i, 'risk_score']:.4f}"
                if i in df_res.index else str(i)
            )
        with cs2:
            if sel in df_res.index:
                st.metric("Risk Score", f"{df_res.loc[sel, 'risk_score']:.4f}")

        if sel in df_res.index:
            row   = df_res.loc[sel]
            score = float(row["risk_score"])

            # banner mức rủi ro
            # ngưỡng phân cấp khớp với risk_level ở Tab 1
            if score >= 0.5:
                css, icon, lbl = "a-hi", "🔴", "RẤT CAO"
                rec = "**Tự động tạm khóa giao dịch** — Gửi OTP cảnh báo khẩn đến chủ thẻ, chuyển hồ sơ sang đội kiểm soát ưu tiên 1."
            elif score >= BEST_THRESHOLD:
                css, icon, lbl = "a-md", "🟡", "TRUNG BÌNH"
                rec = "**Kích hoạt xác thực mạnh (2FA)** — Tăng cường giám sát, đặc biệt nếu danh mục rủi ro cao hoặc giao dịch ban đêm."
            else:
                css, icon, lbl = "a-lo", "🟢", "AN TOÀN"
                rec = "**Tự động phê duyệt** — Tiếp tục monitoring thông thường."

            st.markdown(f"""
            <div class="{css}" style="margin-top:10px">
                <b>{icon} Mức rủi ro: {lbl}</b>
                &nbsp;·&nbsp; Risk Score: <b>{score:.4f}</b>
                &nbsp;·&nbsp; Dự đoán: <b>{"⚠️ Gian lận" if row["fraud_pred"] == 1 else "✅ Hợp lệ"}</b>
            </div>""", unsafe_allow_html=True)

            # gauge + thông tin
            cg, ci = st.columns([1, 2])
            with cg:
                bar_color = "#dc2626" if score >= 0.5 else "#f59e0b" if score >= BEST_THRESHOLD else "#16a34a"
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score * 100,
                    title={"text": "Risk Score (%)", "font": {"size": 13, "color": "#374151"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "#374151"}},
                        "bar":  {"color": bar_color},
                        "steps": [
                            {"range": [0,  16.0],  "color": "#f0fdf4"},
                            {"range": [16.0, 50], "color": "#fffbeb"},
                            {"range": [50, 100], "color": "#fef2f2"},
                        ],
                        "threshold": {
                            "line":  {"color": "#1e40af", "width": 3},
                            "value": threshold * 100,
                        },
                    },
                    number={"suffix": "%", "valueformat": ".1f",
                            "font": {"color": bar_color, "size": 28}},
                ))
                fig_g.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#111827"},
                    height=230,
                    margin=dict(t=20, b=5, l=20, r=20),
                )
                st.plotly_chart(fig_g, use_container_width=True)

            with ci:
                st.markdown('<div class="sec">Thông tin giao dịch</div>', unsafe_allow_html=True)
                info = {
                    "Thời gian":    row.get("trans_date_trans_time", "N/A"),
                    "Danh mục":     row.get("category", "N/A"),
                    "Số tiền":      f"${row.get('amt', 0):,.2f}",
                    "Cửa hàng":     row.get("merchant", "N/A"),
                    "Giới tính":    row.get("gender", "N/A"),
                    "Risk Score":   f"{score:.4f}",
                    "Dự đoán":      "⚠️ Gian lận" if row["fraud_pred"] == 1 else "✅ Hợp lệ",
                }
                for k, v in info.items():
                    st.markdown(f"**{k}:** {v}")

            # ── Yếu tố rủi ro ──────────────────────────
            st.markdown('<div class="sec">Yếu tố ảnh hưởng đến điểm rủi ro</div>',
                        unsafe_allow_html=True)

            risk_f, safe_f = [], []

            amt_ratio = float(row.get("amt_ratio", 1.0))
            if amt_ratio > 5:
                risk_f.append(f"Gấp {amt_ratio:.1f}× chi tiêu trung bình cá nhân")
            elif amt_ratio < 0.5:
                safe_f.append("Thấp hơn mức chi tiêu trung bình")

            if int(row.get("is_night", 0)):
                risk_f.append("Giao dịch ban đêm (22h – 3h)")
            else:
                safe_f.append("Trong giờ giao dịch bình thường")

            if int(row.get("is_online", 0)):
                risk_f.append("Giao dịch trực tuyến (online)")

            v24 = float(row.get("txn_count_24h", 0))
            if v24 >= 5:
                risk_f.append(f"{v24:.0f} giao dịch trong vòng 24h")
            elif v24 <= 1:
                safe_f.append("Tần suất giao dịch bình thường")

            t_gap = float(row.get("time_since_last_txn", 30000))
            if t_gap < 300:
                risk_f.append(f"Giao dịch liên tiếp quá nhanh ({t_gap:.0f}s)")

            dist = float(row.get("distance_km", 75))
            if dist > 130:
                risk_f.append(f"Khoảng cách địa lý bất thường ({dist:.0f} km)")
            elif dist < 10:
                safe_f.append("Giao dịch gần vị trí thường xuyên")

            tier = float(row.get("category_risk_tier", 1))
            if tier == 2:
                risk_f.append(f"Danh mục rủi ro cao ({row.get('category', '')})")
            elif tier == 0:
                safe_f.append(f"Danh mục rủi ro thấp ({row.get('category', '')})")

            amt_val = float(row.get("amt", 0))
            if amt_val > 500:
                risk_f.append(f"Số tiền lớn (${amt_val:,.0f})")

            mf = float(row.get("merchant_freq", 100))
            if mf < 10:
                risk_f.append("Cửa hàng ít gặp trong lịch sử")
            elif mf > 500:
                safe_f.append("Cửa hàng quen thuộc, tần suất cao")

            cr, cs = st.columns(2)
            with cr:
                if risk_f:
                    st.markdown("**↑ Tăng rủi ro**")
                    st.markdown(
                        "".join(f'<span class="chip-r">↑ {r}</span>' for r in risk_f[:6]),
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("*Không phát hiện yếu tố rủi ro đáng kể.*")
            with cs:
                if safe_f:
                    st.markdown("**↓ Giảm rủi ro**")
                    st.markdown(
                        "".join(f'<span class="chip-s">↓ {r}</span>' for r in safe_f[:4]),
                        unsafe_allow_html=True
                    )

            # ── Khuyến nghị ────────────────────────────
            st.markdown('<div class="sec">Khuyến nghị xử lý</div>', unsafe_allow_html=True)
            if score >= 0.5:
                st.error(rec)
            elif score >= BEST_THRESHOLD:
                st.warning(rec)
            else:
                st.success(rec)
