import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, warnings
warnings.filterwarnings("ignore")

# ─── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ─── CSS ───────────────────────────────────────────────────
st.markdown("""<style>
html,body,.stApp,[data-testid="stAppViewContainer"]{background:#f7f8fa!important;color:#111827!important}
[data-testid="stHeader"]{background:#f7f8fa!important}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div:first-child{background:#fff!important;border-right:1px solid #e5e7eb!important}
[data-testid="stSidebar"] *{color:#111827!important}
.block-container{padding-top:1.5rem!important}
[data-testid="stFileUploaderDropzone"]{background:#fff!important;border:1.5px dashed #d1d5db!important;border-radius:8px!important}
[data-testid="stFileUploaderDropzone"] button{background:#2563eb!important;color:#fff!important;border:none!important}
[data-testid="stFileUploaderDropzone"] *{color:#374151!important}
[data-testid="stNumberInput"] input,[data-testid="stNumberInput"] button{background:#fff!important;color:#111827!important;border-color:#d1d5db!important}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:transparent!important}
.stTabs [data-baseweb="tab"]{background:#fff!important;border:1px solid #e5e7eb!important;border-radius:6px 6px 0 0;padding:6px 18px;color:#6b7280!important;font-weight:500}
.stTabs [aria-selected="true"]{background:#2563eb!important;color:#fff!important;border-color:#2563eb!important}
.kpi{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px 20px;text-align:center}
.kpi-val{font-size:1.8rem;font-weight:700;margin:4px 0}
.kpi-lbl{font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em}
.kpi-sub{font-size:.78rem;color:#9ca3af;margin-top:2px}
.sec{font-size:.95rem;font-weight:600;color:#111827;border-left:3px solid #2563eb;padding-left:9px;margin:18px 0 10px}
.a-hi{background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:11px 15px;color:#991b1b}
.a-md{background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:11px 15px;color:#92400e}
.a-lo{background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:11px 15px;color:#166534}
.chip-r{display:inline-block;background:#fef2f2;border:1px solid #fca5a5;border-radius:20px;padding:3px 11px;font-size:.78rem;color:#991b1b;margin:3px}
.chip-s{display:inline-block;background:#f0fdf4;border:1px solid #86efac;border-radius:20px;padding:3px 11px;font-size:.78rem;color:#166534;margin:3px}
</style>""", unsafe_allow_html=True)

# ─── Load model ────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_model():
    with open("fraud_lightgbm_dss.pkl", "rb") as f:
        return pickle.load(f)

payload = load_model()
model        = payload["model"]
FEATURE_COLS = payload["feature_cols"]
CAT_COLS     = payload["categorical_features"]
THRESHOLD    = payload["best_threshold"]
METRICS      = payload["eval_metrics"]
FI_DF        = pd.DataFrame(payload["feature_importance"])

# ─── Feature engineering (khớp pipeline train) ─────────────
CATEGORY_RISK = {
    "shopping_net":2,"misc_net":2,"grocery_pos":2,
    "shopping_pos":1,"gas_transport":1,"misc_pos":1,"travel":1,"entertainment":1,
    "food_dining":0,"health_fitness":0,"personal_care":0,"home":0,
    "kids_pets":0,"grocery_net":0,
}

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # datetime
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # amt_log
    df["amt_log"] = np.log1p(df["amt"].clip(lower=0)) if "amt" in df.columns else 0.0

    # distance
    if all(c in df.columns for c in ["lat","long","merch_lat","merch_long"]):
        def haversine(la1,lo1,la2,lo2):
            R=6371; la1,lo1,la2,lo2=map(np.radians,[la1,lo1,la2,lo2])
            return R*2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
        df["distance_km"] = haversine(df["lat"].fillna(0),df["long"].fillna(0),
                                       df["merch_lat"].fillna(0),df["merch_long"].fillna(0))
    else:
        df.setdefault("distance_km", 75.0)

    # time features
    if "trans_date_trans_time" in df.columns:
        h = df["trans_date_trans_time"].dt.hour.fillna(12).astype(float)
        df["sin_hour"]    = np.sin(2*np.pi*h/24)
        df["cos_hour"]    = np.cos(2*np.pi*h/24)
        df["is_night"]    = ((h>=22)|(h<=3)).astype(int)
    else:
        df["sin_hour"]=df["cos_hour"]=df["is_night"]=0

    # age
    if "dob" in df.columns and "trans_date_trans_time" in df.columns:
        df["age"] = ((df["trans_date_trans_time"]-df["dob"]).dt.days/365.25).fillna(40).clip(18,100)
    else:
        df.setdefault("age", 40.0)

    # category
    if "category" in df.columns:
        df["is_online"]          = df["category"].str.contains("net",na=False).astype(int)
        df["category_risk_tier"] = df["category"].map(CATEGORY_RISK).fillna(1).astype("category")
    else:
        df["is_online"]=0; df["category_risk_tier"]=pd.Categorical([1]*len(df))

    # gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype("category")

    # merchant freq
    df["merchant_freq"] = df["merchant"].map(df["merchant"].value_counts()).fillna(1) \
                          if "merchant" in df.columns else 1.0

    # behavioral (dùng giá trị trong file nếu có, không thì ước lượng)
    if "customer_avg_spending" not in df.columns:
        df["customer_avg_spending"] = df.get("amt", pd.Series([70]*len(df),dtype=float)).mean()
    if "amt_ratio" not in df.columns:
        df["amt_ratio"] = (df["amt"]/df["customer_avg_spending"].clip(lower=1)).clip(0,50) \
                          if "amt" in df.columns else 1.0
    if "time_since_last_txn" not in df.columns:
        df["time_since_last_txn"] = 30000.0
    if "txn_count_24h" not in df.columns:
        df["txn_count_24h"] = 1.0

    # interactions
    df["is_night_x_online"]  = df["is_night"]*df["is_online"]
    df["amt_log_x_is_night"] = df["amt_log"]*df["is_night"]

    return df

def predict(df_raw):
    df_feat = engineer(df_raw)
    missing = [c for c in FEATURE_COLS if c not in df_feat.columns]
    for c in missing:
        df_feat[c] = 0
    X = df_feat[FEATURE_COLS].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    proba = model.predict_proba(X)[:,1]
    pred  = (proba >= THRESHOLD).astype(int)
    return proba, pred

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    st.divider()
    st.markdown("**Cài đặt ngưỡng**")
    threshold = st.slider("Ngưỡng phát hiện", 0.05, 0.95,
                          float(THRESHOLD), 0.05,
                          help=f"Ngưỡng tối ưu từ training: {THRESHOLD}")
    st.divider()
    st.markdown("**Thông tin mô hình**")
    st.markdown(f"- Thuật toán: LightGBM GBDT")
    st.markdown(f"- PR-AUC: **{METRICS.get('pr_auc', 'N/A')}**")
    st.markdown(f"- $-Recall: **{METRICS.get('dollar_recall', 'N/A')}%**")
    st.markdown(f"- Threshold tối ưu: **{THRESHOLD}**")

# ─── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🚨 Phát hiện gian lận", "🔎 Chi tiết giao dịch"])

# ══════════════════════════════════════════════════════════
# TAB 1 — PHÁT HIỆN
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Phát hiện gian lận")

    with st.expander("Xem format CSV yêu cầu"):
        st.markdown("""
Các cột bắt buộc: `amt`, `category`, `gender`, `lat`, `long`, `merch_lat`, `merch_long`  
Cột tùy chọn (nếu có thì kết quả tốt hơn): `trans_date_trans_time`, `dob`, `merchant`,
`customer_avg_spending`, `amt_ratio`, `time_since_last_txn`, `txn_count_24h`
""")

    uploaded = st.file_uploader("Upload file CSV giao dịch", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.caption(f"{len(df_raw):,} giao dịch")

        with st.spinner("Đang dự đoán…"):
            proba, pred = predict(df_raw)

        df_out = df_raw.copy()
        df_out["risk_score"] = np.round(proba, 4)
        df_out["fraud_pred"] = pred
        df_out["risk_level"] = pd.cut(proba, bins=[-0.001,0.3,0.6,1.001],
                                      labels=["Thấp","Trung bình","Cao"])
        # áp threshold tuỳ chỉnh từ sidebar
        df_out["fraud_pred"] = (proba >= threshold).astype(int)
        st.session_state["df_result"] = df_out
        st.session_state["proba"]     = proba

        # KPI
        n       = len(df_out)
        n_fraud = int(df_out["fraud_pred"].sum())
        f_amt   = df_out.loc[df_out["fraud_pred"]==1,"amt"].sum() if "amt" in df_out.columns else 0

        st.markdown('<div class="sec">Tổng kết</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col,(lbl,val,color,sub) in zip([c1,c2,c3,c4],[
            ("Tổng giao dịch", f"{n:,}",           "#2563eb", ""),
            ("Nghi ngờ fraud", f"{n_fraud:,}",      "#dc2626", f"{n_fraud/n*100:.1f}% tổng"),
            ("Thiệt hại ước tính",f"${f_amt:,.0f}", "#d97706", "tổng GD fraud"),
            ("Avg risk score",f"{proba.mean():.3f}","#7c3aed", "toàn tập"),
        ]):
            col.markdown(f"""<div class="kpi">
                <div class="kpi-lbl">{lbl}</div>
                <div class="kpi-val" style="color:{color}">{val}</div>
                <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

        # Charts
        st.markdown('<div class="sec">Phân bố Risk Score</div>', unsafe_allow_html=True)
        ca, cb = st.columns([3,2])
        with ca:
            samp = df_out.sample(min(800,len(df_out)), random_state=1)
            fig = px.scatter(samp, x="amt" if "amt" in samp.columns else samp.index,
                             y="risk_score",
                             color=samp["fraud_pred"].map({0:"Hợp lệ",1:"Fraud"}),
                             color_discrete_map={"Hợp lệ":"#93c5fd","Fraud":"#f87171"},
                             opacity=0.7,
                             labels={"amt":"Số tiền ($)","risk_score":"Risk Score","color":""})
            fig.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"Threshold {threshold}")
            fig.update_layout(template="plotly_white", height=300,
                              margin=dict(t=10,b=30,l=10,r=10), legend=dict(title=""))
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            rc = df_out["risk_level"].value_counts().reset_index()
            rc.columns = ["level","count"]
            fig2 = px.pie(rc, names="level", values="count", hole=0.45,
                          color="level",
                          color_discrete_map={"Thấp":"#86efac","Trung bình":"#fcd34d","Cao":"#f87171"})
            fig2.update_layout(template="plotly_white", height=300,
                               margin=dict(t=10,b=10,l=10,r=10),
                               legend=dict(orientation="h",y=-0.1))
            st.plotly_chart(fig2, use_container_width=True)

        # Feature importance
        st.markdown('<div class="sec">Tầm quan trọng đặc trưng</div>', unsafe_allow_html=True)
        fi_plot = FI_DF.sort_values("gain", ascending=True).tail(12)
        fig3 = px.bar(fi_plot, x="gain", y="feature", orientation="h",
                      color_discrete_sequence=["#2563eb"])
        fig3.update_layout(template="plotly_white", height=300,
                           margin=dict(t=10,b=30,l=10,r=10),
                           xaxis_title="Information Gain", yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

        # Bảng kết quả
        st.markdown('<div class="sec">Danh sách giao dịch</div>', unsafe_allow_html=True)
        f1,f2 = st.columns(2)
        only_fraud = f1.checkbox("Chỉ hiện fraud", value=True)
        min_sc     = f2.slider("Risk score tối thiểu", 0.0, 1.0, float(threshold), 0.05)

        df_show = df_out.copy()
        if only_fraud: df_show = df_show[df_show["fraud_pred"]==1]
        df_show = df_show[df_show["risk_score"] >= min_sc]

        disp = [c for c in ["trans_date_trans_time","category","amt","merchant",
                             "risk_score","risk_level","fraud_pred"] if c in df_show.columns]

        def hl(row):
            s = row.get("risk_score",0)
            if s>=0.6: return ["background-color:#fef2f2"]*len(row)
            if s>=0.3: return ["background-color:#fffbeb"]*len(row)
            return [""]*len(row)

        st.caption(f"{len(df_show):,} giao dịch")
        st.dataframe(df_show[disp].head(500).style.apply(hl,axis=1).format(
            {k:v for k,v in {"risk_score":"{:.4f}","amt":"${:,.2f}"}.items() if k in disp}
        ), use_container_width=True, height=360)

        st.download_button("Tải kết quả CSV", df_out.to_csv(index=False),
                           "fraud_predictions.csv", "text/csv")
    else:
        st.info("Upload file CSV để bắt đầu phát hiện gian lận.")

# ══════════════════════════════════════════════════════════
# TAB 2 — CHI TIẾT
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Chi tiết giao dịch")

    if "df_result" not in st.session_state:
        st.info("Upload và chạy dự đoán ở tab Phát hiện trước.")
    else:
        df_res = st.session_state["df_result"]
        proba_arr = st.session_state["proba"]

        fraud_rows = df_res[df_res["fraud_pred"]==1]
        opts = fraud_rows.index.tolist()[:300] if len(fraud_rows) else df_res.index.tolist()[:300]

        c1,c2 = st.columns([4,1])
        with c1:
            sel = st.selectbox(
                f"Chọn giao dịch ({len(opts)} GD nghi ngờ)",
                options=opts,
                format_func=lambda i: (
                    f"#{i}  "
                    f"{df_res.loc[i,'category'] if 'category' in df_res.columns else ''}  "
                    f"— ${df_res.loc[i,'amt']:.2f}  "
                    f"— score {df_res.loc[i,'risk_score']:.3f}"
                ) if i in df_res.index else str(i)
            )
        with c2:
            if sel in df_res.index:
                st.metric("Risk Score", f"{df_res.loc[sel,'risk_score']:.4f}")

        if sel in df_res.index:
            row   = df_res.loc[sel]
            score = float(row["risk_score"])

            if score>=0.6:   css,icon,lbl = "a-hi","🔴","CAO"
            elif score>=0.3: css,icon,lbl = "a-md","🟡","TRUNG BÌNH"
            else:            css,icon,lbl = "a-lo","🟢","THẤP"

            st.markdown(f"""<div class="{css}" style="margin-top:10px">
                <b>{icon} Mức rủi ro: {lbl}</b> &nbsp;·&nbsp;
                Score: <b>{score:.4f}</b> &nbsp;·&nbsp;
                Dự đoán: <b>{"⚠️ Gian lận" if row['fraud_pred']==1 else "✅ Hợp lệ"}</b>
            </div>""", unsafe_allow_html=True)

            cg, ci = st.columns([1,2])
            with cg:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=score*100,
                    title={"text":"Risk (%)","font":{"size":14}},
                    gauge={
                        "axis":{"range":[0,100]},
                        "bar":{"color":"#dc2626" if score>0.6 else "#f59e0b" if score>0.3 else "#16a34a"},
                        "steps":[{"range":[0,30],"color":"#f0fdf4"},
                                  {"range":[30,60],"color":"#fffbeb"},
                                  {"range":[60,100],"color":"#fef2f2"}],
                        "threshold":{"line":{"color":"#1e40af","width":3},"value":threshold*100},
                    },
                    number={"suffix":"%","valueformat":".1f"},
                ))
                fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                    font={"color":"#111827"},
                                    height=220, margin=dict(t=20,b=5,l=20,r=20))
                st.plotly_chart(fig_g, use_container_width=True)

            with ci:
                st.markdown('<div class="sec">Thông tin giao dịch</div>', unsafe_allow_html=True)
                for k,v in {
                    "Thời gian": row.get("trans_date_trans_time","N/A"),
                    "Danh mục":  row.get("category","N/A"),
                    "Số tiền":   f"${row.get('amt',0):,.2f}",
                    "Cửa hàng":  row.get("merchant","N/A"),
                    "Giới tính": row.get("gender","N/A"),
                }.items():
                    st.markdown(f"**{k}:** {v}")

            # Yếu tố rủi ro (dựa trực tiếp trên feature value)
            st.markdown('<div class="sec">Yếu tố rủi ro</div>', unsafe_allow_html=True)
            risk_factors, safe_factors = [], []

            amt_ratio = float(row.get("amt_ratio",1))
            if amt_ratio>5:   risk_factors.append(f"Gấp {amt_ratio:.1f}× chi tiêu trung bình")
            elif amt_ratio<0.5: safe_factors.append("Dưới mức chi tiêu trung bình")

            if int(row.get("is_night",0)): risk_factors.append("Giao dịch ban đêm (22h–3h)")
            else: safe_factors.append("Giờ giao dịch bình thường")

            if int(row.get("is_online",0)): risk_factors.append("Giao dịch online")

            v24 = float(row.get("txn_count_24h",0))
            if v24>=5:   risk_factors.append(f"{v24:.0f} giao dịch trong 24h")
            elif v24<=1: safe_factors.append("Tần suất giao dịch bình thường")

            t_gap = float(row.get("time_since_last_txn",30000))
            if t_gap<300: risk_factors.append(f"Giao dịch liên tiếp ({t_gap:.0f}s)")

            dist = float(row.get("distance_km",75))
            if dist>130: risk_factors.append(f"Khoảng cách xa ({dist:.0f} km)")

            tier = float(row.get("category_risk_tier",1))
            if tier==2:   risk_factors.append(f"Danh mục rủi ro cao")
            elif tier==0: safe_factors.append("Danh mục rủi ro thấp")

            cr, cs = st.columns(2)
            with cr:
                if risk_factors:
                    st.markdown("**Tăng rủi ro**")
                    st.markdown("".join(f'<span class="chip-r">↑ {r}</span>' for r in risk_factors[:5]),
                                unsafe_allow_html=True)
            with cs:
                if safe_factors:
                    st.markdown("**Giảm rủi ro**")
                    st.markdown("".join(f'<span class="chip-s">↓ {r}</span>' for r in safe_factors[:3]),
                                unsafe_allow_html=True)

            # Khuyến nghị
            st.markdown('<div class="sec">Khuyến nghị</div>', unsafe_allow_html=True)
            if score>=0.6:
                st.error("**Tạm giữ giao dịch** — Yêu cầu xác thực bổ sung từ chủ thẻ.")
            elif score>=0.3:
                st.warning("**Theo dõi** — Ghi nhận, yêu cầu 2FA cho giao dịch tiếp theo.")
            else:
                st.success("**Cho phép** — Rủi ro thấp.")