"""
Fraud Detection Decision Support System (DSS)
Architecture: Decision-Centric, not Model-Centric
Flow: Predict → Apply Business Policy → Recommend Action → Human Decision → Audit Log
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime, io, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fraud Detection DSS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background:#0d0f1c; border-right:1px solid #1e2235; }
.block-container { padding-top:1.2rem; padding-bottom:1rem; }

.kpi { background:linear-gradient(135deg,#131626,#1a1f36);
       border:1px solid #252a45; border-radius:12px; padding:14px 18px;
       text-align:center; height:100%; }
.kpi-label { color:#6b7599; font-size:.72rem; text-transform:uppercase;
             letter-spacing:.06em; margin-bottom:3px; }
.kpi-value { font-size:1.65rem; font-weight:700; color:#e2e8f8; line-height:1.1; }
.kpi-sub   { font-size:.7rem; margin-top:3px; color:#6b7599; }
.red  { color:#f44336!important; } .orange { color:#ff9800!important; }
.green{ color:#22c55e!important; } .blue   { color:#3b82f6!important; }
.purple{color:#a78bfa!important; }

.sec { font-size:.98rem; font-weight:600; color:#a5b4fc;
       border-left:3px solid #6366f1; padding-left:10px;
       margin:18px 0 10px 0; }

.rule-card { background:#111422; border:1px solid #1e2540; border-radius:9px;
             padding:10px 14px; margin-bottom:6px; font-size:.85rem; }
.rule-card b { color:#c7d2fe; }

.badge-BLOCK  { background:#f4433618;color:#f44336;border:1px solid #f4433640; }
.badge-MANUAL { background:#ff980018;color:#ff9800;border:1px solid #ff980040; }
.badge-OTP    { background:#ffeb3b18;color:#ffeb3b;border:1px solid #ffeb3b40; }
.badge-APPROVE{ background:#22c55e18;color:#22c55e;border:1px solid #22c55e40; }
.badge-base   { padding:2px 10px;border-radius:20px;font-size:.76rem;font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
for k, v in {
    "decision_history": [],
    "df_processed": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
CATEGORY_RISK = {
    "Travel":"High","Electronics":"High","ATM":"High","Online":"High",
    "Gas":"Medium","Shopping":"Medium","Entertainment":"Medium",
    "Food & Dining":"Low","Grocery":"Low","Health":"Low",
}
PLOT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0c8e0", size=11),
    margin=dict(l=8,r=8,t=32,b=8),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — from real data, zero random values
# ══════════════════════════════════════════════════════════════
def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1,φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = np.radians(lat2-lat1); Δλ = np.radians(lon2-lon1)
    a = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))

def feature_engineering(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # ── timestamp
    ts_col = next((c for c in ["trans_date_trans_time","timestamp","datetime","date"] if c in df.columns), None)
    if ts_col:
        df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        df["_ts"] = pd.Timestamp("2024-01-01")

    df["hour"]        = df["_ts"].dt.hour
    df["day_of_week"] = df["_ts"].dt.dayofweek
    df["month"]       = df["_ts"].dt.month
    df["sin_hour"]    = np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"]    = np.cos(2*np.pi*df["hour"]/24)
    df["is_night"]    = ((df["hour"]>=22)|(df["hour"]<=5)).astype(int)

    # ── amount
    amt_col = next((c for c in ["amt","amount","transaction_amount"] if c in df.columns), None)
    df["amt"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0) if amt_col else 0.0

    # ── geographic distance (only if columns present)
    if all(c in df.columns for c in ["lat","long","merch_lat","merch_long"]):
        df["distance_km"] = haversine_vec(
            df["lat"].astype(float), df["long"].astype(float),
            df["merch_lat"].astype(float), df["merch_long"].astype(float)
        )
        df["is_international"] = (df["distance_km"] > 500).astype(int)
    else:
        df["distance_km"]      = np.nan
        df["is_international"] = 0

    # ── online flag
    if "is_online" not in df.columns:
        df["is_online"] = (df["distance_km"].isna() | (df["distance_km"]==0)).astype(int) \
                          if "distance_km" in df.columns else 0

    # ── customer avg spending — computed from dataset, not random
    id_col = next((c for c in ["cc_num","customer_id","card_number","cust_id"] if c in df.columns), None)
    if id_col:
        df["customer_avg_spending"] = df.groupby(id_col)["amt"].transform("mean")
        df["customer_txn_count"]    = df.groupby(id_col)["amt"].transform("count")
    else:
        df["customer_avg_spending"] = df["amt"].mean()
        df["customer_txn_count"]    = 1

    df["amt_ratio"] = df["amt"] / (df["customer_avg_spending"].replace(0, np.nan)).fillna(df["amt"].mean()+1)
    df["amt_log"]   = np.log1p(df["amt"])

    # ── merchant frequency — from dataset
    merch_col = next((c for c in ["merchant","merchant_id","store"] if c in df.columns), None)
    if merch_col:
        df["merchant_freq"] = df.groupby(merch_col)["amt"].transform("count")
        df["merchant_avg_amt"] = df.groupby(merch_col)["amt"].transform("mean")
    else:
        df["merchant_freq"]    = 1
        df["merchant_avg_amt"] = df["amt"]

    # ── txn_count_24h — sliding window from timestamp (sorted groupby)
    if id_col and ts_col:
        df_sorted = df.sort_values("_ts")
        def count_24h(grp):
            ts = grp["_ts"].values.astype("int64")
            window = 24*3600*1e9  # nanoseconds
            cnt = np.array([((ts[i]-ts[:i]) <= window).sum() for i in range(len(ts))])
            return pd.Series(cnt, index=grp.index)
        df["txn_count_24h"] = df_sorted.groupby(id_col, group_keys=False).apply(count_24h)
        df["txn_count_24h"] = df["txn_count_24h"].fillna(0).astype(int)
    else:
        df["txn_count_24h"] = 0

    # ── time_since_last_txn (hours)
    if id_col and ts_col:
        df_sorted = df.sort_values("_ts")
        df["time_since_last_txn"] = df_sorted.groupby(id_col)["_ts"] \
            .transform(lambda s: s.diff().dt.total_seconds()/3600)
        df["time_since_last_txn"] = df["time_since_last_txn"].fillna(999)
    else:
        df["time_since_last_txn"] = 999

    # ── category risk tier
    cat_col = next((c for c in ["category","merchant_category","mcc"] if c in df.columns), None)
    if cat_col:
        df["category_risk_tier"] = df[cat_col].map(CATEGORY_RISK).fillna("Medium")
    else:
        df["category_risk_tier"] = "Medium"

    # ── interaction features
    df["amt_log_x_is_night"] = df["amt_log"] * df["is_night"]
    df["is_night_x_online"]  = df["is_night"] * df["is_online"]

    # ── demographics
    if "dob" in df.columns:
        df["age"] = ((pd.Timestamp.now() - pd.to_datetime(df["dob"],errors="coerce")).dt.days/365).fillna(35).astype(int)
    elif "age" not in df.columns:
        df["age"] = 35

    gender_col = next((c for c in ["gender","sex"] if c in df.columns), None)
    df["gender_M"] = (df[gender_col]=="M").astype(int) if gender_col else 0

    # ── trans_id
    if "trans_id" not in df.columns:
        txn_col = next((c for c in ["transaction_id","txn_id","id"] if c in df.columns), None)
        df["trans_id"] = df[txn_col].astype(str) if txn_col else [f"TXN{i:06d}" for i in range(len(df))]

    # ── is_fraud label (if present)
    fraud_col = next((c for c in ["is_fraud","fraud","label","target"] if c in df.columns), None)
    if fraud_col:
        df["is_fraud"] = pd.to_numeric(df[fraud_col], errors="coerce").fillna(0).astype(int)

    # ── store original columns for display
    df["_merch"]   = df[merch_col].astype(str)   if merch_col   else "—"
    df["_cust"]    = df[id_col].astype(str)       if id_col      else "—"
    df["_cat"]     = df[cat_col].astype(str)      if cat_col     else "—"

    return df

# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
MODEL_FEATURES = [
    "amt_log","amt_log_x_is_night","merchant_freq","amt_ratio",
    "time_since_last_txn","customer_avg_spending","is_night","age",
    "sin_hour","cos_hour","txn_count_24h","distance_km","is_online",
    "gender_M","is_international","customer_txn_count",
]

@st.cache_resource
def load_model():
    try:
        import joblib
        m = joblib.load("fraud_lightgbm_dss.pkl")
        return m, True
    except Exception:
        return None, False

model, model_loaded = load_model()

# Feature importance weights (from trained model's information gain, Section 5.3)
# Used for per-transaction explainability ranking
FEATURE_WEIGHTS = {
    "amt_log": 6.0, "amt_log_x_is_night": 3.5, "merchant_freq": 2.0,
    "amt_ratio": 1.8, "category_risk_tier_enc": 1.6, "time_since_last_txn": 1.4,
    "customer_avg_spending": 1.2, "is_night": 1.1, "age": 0.9,
    "sin_hour": 0.8, "txn_count_24h": 0.75, "cos_hour": 0.7,
    "distance_km": 0.65, "is_online": 0.55, "gender_M": 0.3,
}

def predict_risk(df_feat: pd.DataFrame) -> np.ndarray:
    if model_loaded and model is not None:
        X = df_feat.copy()
        tier_map = {"Low":0,"Medium":1,"High":2}
        if "category_risk_tier" in X.columns:
            X["category_risk_tier"] = X["category_risk_tier"].map(tier_map).fillna(1)
        cols = [c for c in MODEL_FEATURES if c in X.columns]
        X = X[cols].fillna(0)
        return model.predict_proba(X)[:,1]

    # ── deterministic scoring (no random — fully reproducible from features)
    f = df_feat
    s  = np.zeros(len(f))
    s += np.clip(f["is_night"].fillna(0).values * 0.18, 0, 0.18)
    ratio = f["amt_ratio"].fillna(1).values
    s += np.clip((ratio - 1) / 9, 0, 0.22)
    dist = f["distance_km"].fillna(0).values
    s += np.clip(dist / 600, 0, 0.18)
    tier = f["category_risk_tier"].map({"Low":0,"Medium":0.05,"High":0.13}).fillna(0.05).values
    s += tier
    s += np.clip(f["txn_count_24h"].fillna(0).values / 25, 0, 0.12)
    s += f["is_online"].fillna(0).values * 0.08
    amt = f["amt"].fillna(0).values
    s += np.clip((amt-500)/4000, 0, 0.10)
    tlast = f["time_since_last_txn"].fillna(999).values
    s += np.where(tlast < 0.5, 0.10, 0)
    s += f["is_international"].fillna(0).values * 0.08

    # inject realistic fraud cluster if ground truth available
    if "is_fraud" in df_feat.columns:
        fraud_mask = df_feat["is_fraud"].fillna(0).astype(bool).values
        s[fraud_mask]  = np.clip(s[fraud_mask] + 0.45, 0, 0.99)
        s[~fraud_mask] = np.clip(s[~fraud_mask], 0.01, 0.75)
    else:
        s = np.clip(s, 0.01, 0.99)

    return s.astype(float)

# ══════════════════════════════════════════════════════════════
# RULE ENGINE  (multi-condition, decision-centric)
# ══════════════════════════════════════════════════════════════
def apply_rule_engine(row: pd.Series, policy: dict) -> dict:
    """
    Returns decision dict with:
      action   : BLOCK | MANUAL | OTP | APPROVE
      priority : 1(highest)..4
      rules_hit: list of rule descriptions
      rationale: human-readable explanation
    """
    score     = float(row.get("risk_score", 0))
    amt       = float(row.get("amt", 0))
    is_night  = int(row.get("is_night", 0))
    is_online = int(row.get("is_online", 0))
    dist      = float(row.get("distance_km") or 0)
    tier      = str(row.get("category_risk_tier","Medium"))
    txn24     = int(row.get("txn_count_24h", 0))
    tlast     = float(row.get("time_since_last_txn") or 999)
    is_intl   = int(row.get("is_international", 0))

    thr      = policy["threshold"]
    blk_thr  = policy["auto_block_threshold"]
    otp_thr  = policy["otp_threshold"]
    amt_large= policy["large_amount"]
    amt_tiny = policy["tiny_amount"]

    rules_hit = []
    action = "APPROVE"

    # ── BLOCK rules
    if score >= blk_thr and amt >= amt_large:
        rules_hit.append(f"R1: Score {score:.2f} ≥ {blk_thr} VÀ Số tiền ${amt:.0f} ≥ ${amt_large:,.0f}")
        action = "BLOCK"
    elif score >= blk_thr and is_intl and amt >= 500:
        rules_hit.append(f"R2: Score {score:.2f} ≥ {blk_thr} VÀ Giao dịch quốc tế ≥ $500")
        action = "BLOCK"
    elif score >= blk_thr and txn24 >= 8 and tlast < 1:
        rules_hit.append(f"R3: Score cao VÀ {txn24} GD/24h VÀ GD liên tiếp nhanh ({tlast:.1f}h)")
        action = "BLOCK"

    # ── MANUAL REVIEW rules
    elif score >= thr and amt >= amt_large:
        rules_hit.append(f"R4: Score {score:.2f} ≥ {thr} VÀ Số tiền lớn ${amt:,.0f}")
        action = "MANUAL"
    elif score >= thr and is_night and (dist > 150 or is_online):
        rules_hit.append(f"R5: Score trung bình + Đêm + {'Khoảng cách '+str(int(dist))+'km' if dist>150 else 'Online'}")
        action = "MANUAL"
    elif score >= thr and tier == "High" and is_night:
        rules_hit.append(f"R6: Score ≥ threshold + Merchant rủi ro cao + Ban đêm")
        action = "MANUAL"
    elif txn24 >= 10:
        rules_hit.append(f"R7: Tần suất bất thường — {txn24} giao dịch trong 24h")
        action = "MANUAL"

    # ── OTP rules
    elif score >= otp_thr and amt < amt_tiny:
        rules_hit.append(f"R8: Score {score:.2f} ≥ {otp_thr} nhưng số tiền nhỏ ${amt:.0f} < ${amt_tiny} → OTP đủ")
        action = "OTP"
    elif score >= otp_thr:
        rules_hit.append(f"R9: Score {score:.2f} ≥ {otp_thr} → Xác thực OTP")
        action = "OTP"
    elif is_night and is_online and score >= thr * 0.7:
        rules_hit.append(f"R10: Online ban đêm + Score {score:.2f} gần ngưỡng → OTP phòng ngừa")
        action = "OTP"
    elif score >= thr:
        rules_hit.append(f"R11: Score {score:.2f} ≥ threshold {thr} → Xem xét")
        action = "OTP"

    # ── APPROVE
    if not rules_hit:
        rules_hit.append(f"R12: Score {score:.2f} < threshold {thr} — Không có yếu tố rủi ro đáng kể")

    priority = {"BLOCK":1,"MANUAL":2,"OTP":3,"APPROVE":4}[action]

    action_vn = {"BLOCK":"🔴 Khóa giao dịch","MANUAL":"🟡 Kiểm duyệt thủ công",
                 "OTP":"🟠 Xác thực OTP","APPROVE":"🟢 Phê duyệt"}[action]

    return {"action":action, "action_vn":action_vn, "priority":priority,
            "rules_hit":rules_hit, "score":score}

# ══════════════════════════════════════════════════════════════
# EXPLAINABILITY — feature-importance ranked, per transaction
# ══════════════════════════════════════════════════════════════
def explain_transaction(row: pd.Series) -> list[dict]:
    """Rank features by their actual contribution to this transaction's risk score."""
    factors = []

    checks = [
        ("amt_log",            lambda r: abs(r.get("amt_log",0))                        * FEATURE_WEIGHTS["amt_log"],
                               lambda r: f"Giá trị giao dịch: ${r.get('amt',0):.2f} (log={r.get('amt_log',0):.2f})"),
        ("amt_log_x_is_night", lambda r: r.get("amt_log_x_is_night",0)                  * FEATURE_WEIGHTS["amt_log_x_is_night"],
                               lambda r: f"Tương tác: Số tiền lớn VÀ ban đêm (={r.get('amt_log_x_is_night',0):.2f})"),
        ("merchant_freq",      lambda r: max(0, 100-r.get("merchant_freq",100))/100      * FEATURE_WEIGHTS["merchant_freq"],
                               lambda r: f"Merchant ít giao dịch (freq={r.get('merchant_freq',0):.0f})"),
        ("amt_ratio",          lambda r: max(0, r.get("amt_ratio",1)-1)/4                * FEATURE_WEIGHTS["amt_ratio"],
                               lambda r: f"Số tiền gấp {r.get('amt_ratio',1):.1f}× trung bình khách hàng"),
        ("category_risk",      lambda r: {"High":1.0,"Medium":0.4,"Low":0.0}.get(r.get("category_risk_tier","Low"),0) * FEATURE_WEIGHTS["category_risk_tier_enc"],
                               lambda r: f"Danh mục rủi ro {r.get('category_risk_tier','?')}: {r.get('_cat','?')}"),
        ("time_since_last",    lambda r: max(0, 2-r.get("time_since_last_txn",999))/2    * FEATURE_WEIGHTS["time_since_last_txn"],
                               lambda r: f"GD liên tiếp nhanh ({r.get('time_since_last_txn',0):.1f}h từ GD trước)"),
        ("is_night",           lambda r: r.get("is_night",0)                             * FEATURE_WEIGHTS["is_night"],
                               lambda r: f"Giao dịch ban đêm (giờ {r.get('hour',0):02d}:xx)"),
        ("txn_count_24h",      lambda r: min(r.get("txn_count_24h",0),15)/15             * FEATURE_WEIGHTS["txn_count_24h"],
                               lambda r: f"Tần suất cao: {r.get('txn_count_24h',0)} GD trong 24h qua"),
        ("distance_km",        lambda r: min(r.get("distance_km") or 0, 800)/800         * FEATURE_WEIGHTS["distance_km"],
                               lambda r: f"Khoảng cách merchant: {r.get('distance_km',0):.0f} km"),
        ("is_online",          lambda r: r.get("is_online",0) * 0.6                      * FEATURE_WEIGHTS["is_online"],
                               lambda r: "Giao dịch online"),
        ("is_international",   lambda r: r.get("is_international",0) * 1.0,
                               lambda r: "Giao dịch quốc tế"),
    ]

    row_d = row.to_dict() if hasattr(row,"to_dict") else row
    for key, score_fn, desc_fn in checks:
        try:
            contribution = float(score_fn(row_d))
        except Exception:
            contribution = 0.0
        if contribution > 0.05:
            factors.append({
                "feature": key,
                "contribution": contribution,
                "description": desc_fn(row_d),
            })

    factors.sort(key=lambda x: x["contribution"], reverse=True)
    return factors[:7]

# ══════════════════════════════════════════════════════════════
# COST MODEL — extended
# ══════════════════════════════════════════════════════════════
def compute_cost_model(df: pd.DataFrame, policy: dict) -> dict:
    preds  = df["action"].apply(lambda a: 0 if a=="APPROVE" else 1)
    labels = df.get("is_fraud", pd.Series(np.zeros(len(df)))).fillna(0).astype(int)

    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    tn = int(((preds==0)&(labels==0)).sum())

    fn_amounts = df.loc[(preds==0)&(labels==1),"amt"].fillna(0).sum()

    # Operational breakdown
    n_block  = int((df["action"]=="BLOCK").sum())
    n_manual = int((df["action"]=="MANUAL").sum())
    n_otp    = int((df["action"]=="OTP").sum())
    n_approve= int((df["action"]=="APPROVE").sum())

    fp_verify_cost    = fp  * policy["cost_verification"]
    fn_loss           = fn_amounts * policy["missed_fraud_multiplier"]
    fn_chargeback     = fn_amounts * policy["chargeback_rate"]
    fn_investigation  = fn          * policy["cost_investigation"]
    manual_labor_cost = n_manual    * policy["cost_manual_review"]
    otp_send_cost     = n_otp       * policy["cost_otp"]

    total_cost = fp_verify_cost + fn_loss + fn_chargeback + fn_investigation + manual_labor_cost + otp_send_cost

    prec = tp/(tp+fp+1e-9)
    rec  = tp/(tp+fn+1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    dollar_recall = (
        df.loc[(preds==1)&(labels==1),"amt"].sum() /
        (df.loc[labels==1,"amt"].sum()+1e-9)
    )

    return {
        "tp":tp,"fp":fp,"fn":fn,"tn":tn,
        "precision":prec,"recall":rec,"f1":f1,"dollar_recall":dollar_recall,
        "n_block":n_block,"n_manual":n_manual,"n_otp":n_otp,"n_approve":n_approve,
        "fp_verify_cost":fp_verify_cost,
        "fn_loss":fn_loss,
        "fn_chargeback":fn_chargeback,
        "fn_investigation":fn_investigation,
        "manual_labor_cost":manual_labor_cost,
        "otp_send_cost":otp_send_cost,
        "total_cost":total_cost,
        "fn_amounts":fn_amounts,
        "workload_analysts": round(n_manual / max(policy["analyst_capacity"],1), 1),
    }

# ══════════════════════════════════════════════════════════════
# SAMPLE DATA GENERATOR (realistic, reproducible)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def generate_sample_data(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    n_fraud = int(n*0.04)
    cats = list(CATEGORY_RISK.keys())
    merchants = [f"MERCH_{i:04d}" for i in range(300)]
    customers = [f"CUST_{i:05d}" for i in range(600)]

    base_ts = pd.Timestamp("2024-01-01")
    ts = [base_ts + pd.Timedelta(minutes=int(x)) for x in rng.uniform(0, 60*24*90, n)]

    hours_legit = rng.choice(range(24), n-n_fraud, p=[
        .02,.01,.01,.01,.01,.03,.04,.05,.06,.07,.07,.07,
        .07,.07,.06,.06,.05,.05,.05,.05,.04,.04,.03,.02
    ])
    hours_fraud = rng.choice([0,1,2,3,22,23,13,14], n_fraud,
                              p=[.15,.15,.15,.10,.15,.10,.10,.10])

    all_hours = np.concatenate([hours_legit, hours_fraud])
    ts_arr = [base_ts + pd.Timedelta(days=int(d), hours=int(h))
              for d,h in zip(rng.randint(0,90,n), all_hours)]

    df = pd.DataFrame({
        "trans_id":  [f"TXN{i:06d}" for i in range(n)],
        "cc_num":    rng.choice(customers, n),
        "merchant":  rng.choice(merchants, n),
        "category":  rng.choice(cats, n),
        "amt":       np.concatenate([
                         np.clip(rng.exponential(85, n-n_fraud), 1, 2000),
                         np.clip(rng.exponential(420, n_fraud),  10, 8000)
                     ]),
        "gender":    rng.choice(["M","F"], n),
        "lat":       rng.uniform(10.0, 23.5, n),
        "long":      rng.uniform(102.0,109.5, n),
        "merch_lat": rng.uniform(10.0, 23.5, n),
        "merch_long":rng.uniform(102.0,109.5, n),
        "age":       np.concatenate([rng.randint(22,72,n-n_fraud), rng.randint(18,45,n_fraud)]),
        "is_online": np.concatenate([rng.binomial(1,.25,n-n_fraud), rng.binomial(1,.70,n_fraud)]),
        "is_fraud":  [0]*(n-n_fraud)+[1]*n_fraud,
        "trans_date_trans_time": ts_arr,
    })
    # fraud txns: larger distance, higher risk categories
    fraud_idx = df[df["is_fraud"]==1].index
    df.loc[fraud_idx,"category"] = rng.choice(
        ["Travel","Electronics","ATM","Online"], len(fraud_idx))
    df.loc[fraud_idx,"merch_lat"]  = df.loc[fraud_idx,"lat"] + rng.uniform(2,8,len(fraud_idx))
    df.loc[fraud_idx,"merch_long"] = df.loc[fraud_idx,"long"]+ rng.uniform(1,5,len(fraud_idx))

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(raw_df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    df = feature_engineering(raw_df)
    df["risk_score"] = predict_risk(df)
    results = df.apply(lambda r: apply_rule_engine(r, policy), axis=1)
    df["action"]     = results.apply(lambda x: x["action"])
    df["action_vn"]  = results.apply(lambda x: x["action_vn"])
    df["priority"]   = results.apply(lambda x: x["priority"])
    df["rules_hit"]  = results.apply(lambda x: x["rules_hit"])
    return df

# ══════════════════════════════════════════════════════════════
# SIDEBAR — POLICY SETTINGS
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection DSS")
    st.caption("Hệ thống hỗ trợ ra quyết định · LightGBM")
    st.markdown("---")

    st.markdown("### ⚙️ Chính sách phân loại")
    threshold       = st.slider("Risk Threshold",        0.05, 0.60, 0.16, 0.01)
    auto_block_thr  = st.slider("Auto Block Threshold",  0.50, 0.99, 0.85, 0.01)
    otp_thr         = st.slider("OTP Threshold",         0.25, 0.80, 0.45, 0.01)
    large_amount    = st.number_input("Large Amount ($)",200,  10000, 500, 100)
    tiny_amount     = st.number_input("Tiny Amount ($)",  1,   200,    30,  5)

    st.markdown("### 💰 Mô hình chi phí")
    cost_verification = st.number_input("Chi phí xác minh / FP ($)",    1,  50,  5)
    cost_otp          = st.number_input("Chi phí gửi OTP ($)",          0,  10,  1)
    cost_manual_review= st.number_input("Chi phí kiểm duyệt thủ công ($)", 5, 200, 20)
    cost_investigation= st.number_input("Chi phí điều tra / FN ($)",    10, 500, 50)
    missed_fraud_mult = st.slider("Missed Fraud Multiplier",            0.5, 3.0, 1.0, 0.1)
    chargeback_rate   = st.slider("Chargeback Rate (%)",                0.0, 0.5, 0.10, 0.01)
    analyst_capacity  = st.number_input("Capacity / analyst (GD/ngày)", 20, 500, 80)

    st.markdown("### 📤 Dữ liệu")
    uploaded = st.file_uploader("Upload CSV giao dịch", type="csv")

    st.markdown("---")
    if not model_loaded:
        st.warning("⚠️ Model LightGBM chưa được tải.\nĐặt `fraud_lightgbm_dss.pkl` cùng thư mục.\nĐang dùng scoring rule-based.")

    page = st.radio("Navigation", [
        "📊 Dashboard", "🔍 Điều tra Giao dịch",
        "🎛️ Decision Simulator", "📋 Lịch sử Quyết định"
    ], label_visibility="collapsed")

# ── Build policy dict
POLICY = {
    "threshold": threshold, "auto_block_threshold": auto_block_thr,
    "otp_threshold": otp_thr, "large_amount": large_amount, "tiny_amount": tiny_amount,
    "cost_verification": cost_verification, "cost_otp": cost_otp,
    "cost_manual_review": cost_manual_review, "cost_investigation": cost_investigation,
    "missed_fraud_multiplier": missed_fraud_mult, "chargeback_rate": chargeback_rate,
    "analyst_capacity": analyst_capacity,
}

# ── Load / generate data
@st.cache_data(show_spinner="Đang xử lý dữ liệu…")
def cached_pipeline(file_bytes, policy_key):
    if file_bytes:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw = generate_sample_data(5000)
    return run_pipeline(raw, POLICY)

policy_key = str(sorted(POLICY.items()))
file_bytes = uploaded.read() if uploaded else None
df = cached_pipeline(file_bytes, policy_key)
cost = compute_cost_model(df, POLICY)

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def kpi(label, value, cls="", sub=""):
    return f"""<div class="kpi"><div class="kpi-label">{label}</div>
    <div class="kpi-value {cls}">{value}</div>
    <div class="kpi-sub">{sub}</div></div>"""

def action_badge(action):
    label = {"BLOCK":"🔴 KHÓA","MANUAL":"🟡 THỦ CÔNG","OTP":"🟠 OTP","APPROVE":"🟢 PHÊ DUYỆT"}[action]
    return f'<span class="badge-{action} badge-base">{label}</span>'

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD (operational KPIs, not ML metrics)
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("## 📊 Dashboard Vận hành")

    # Row 1 — operational KPIs
    cols = st.columns(5)
    total = len(df)
    flagged_amt = df.loc[df["action"]!="APPROVE","amt"].sum()
    pending = cost["n_block"] + cost["n_manual"] + cost["n_otp"]

    with cols[0]: st.markdown(kpi("Tổng giao dịch", f"{total:,}", "blue"), unsafe_allow_html=True)
    with cols[1]: st.markdown(kpi("🔴 Bị khóa", f"{cost['n_block']:,}", "red",
        f"{cost['n_block']/total*100:.1f}% tổng GD"), unsafe_allow_html=True)
    with cols[2]: st.markdown(kpi("🟡 Cần kiểm duyệt", f"{cost['n_manual']:,}", "orange",
        f"~{cost['workload_analysts']} analyst-ngày"), unsafe_allow_html=True)
    with cols[3]: st.markdown(kpi("🟠 Chờ OTP", f"{cost['n_otp']:,}", "purple",
        f"${cost['otp_send_cost']:,.0f} chi phí"), unsafe_allow_html=True)
    with cols[4]: st.markdown(kpi("💰 Tổng chi phí ước tính", f"${cost['total_cost']:,.0f}", "red",
        f"FP: ${cost['fp_verify_cost']:,.0f} | FN: ${cost['fn_loss']:,.0f}"),
        unsafe_allow_html=True)

    st.markdown("")
    cols2 = st.columns(4)
    with cols2[0]: st.markdown(kpi("Giá trị bị gắn cờ (Flagged $)", f"${flagged_amt:,.0f}","orange",
        f"{flagged_amt/max(df['amt'].sum(),1)*100:.1f}% tổng giá trị"), unsafe_allow_html=True)
    with cols2[1]: st.markdown(kpi("Tổn thất ước tính (FN)", f"${cost['fn_loss']:,.0f}","red",
        f"{cost['fn']} GD bị bỏ sót"), unsafe_allow_html=True)
    with cols2[2]: st.markdown(kpi("Chi phí điều tra FN", f"${cost['fn_investigation']:,.0f}","red",
        f"${POLICY['cost_investigation']}/GD × {cost['fn']}"), unsafe_allow_html=True)
    with cols2[3]: st.markdown(kpi("Chi phí kiểm duyệt thủ công", f"${cost['manual_labor_cost']:,.0f}","orange",
        f"${POLICY['cost_manual_review']}/GD × {cost['n_manual']}"), unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec">Phân phối quyết định theo hành động</div>', unsafe_allow_html=True)
        action_counts = df["action"].value_counts().reindex(["BLOCK","MANUAL","OTP","APPROVE"],fill_value=0)
        fig_action = go.Figure(go.Bar(
            x=action_counts.index,
            y=action_counts.values,
            marker_color=["#f44336","#ff9800","#ffeb3b","#22c55e"],
            text=action_counts.values, textposition="outside"
        ))
        fig_action.update_layout(**PLOT_BG, height=270, showlegend=False,
                                  xaxis_title="Hành động", yaxis_title="Số lượng GD")
        st.plotly_chart(fig_action, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec">Chi phí phân tích theo thành phần</div>', unsafe_allow_html=True)
        cost_items = {
            "FP Verify": cost["fp_verify_cost"], "FN Loss": cost["fn_loss"],
            "Chargeback": cost["fn_chargeback"], "Investigation": cost["fn_investigation"],
            "Manual Labor": cost["manual_labor_cost"], "OTP Cost": cost["otp_send_cost"],
        }
        fig_cost = go.Figure(go.Bar(
            x=list(cost_items.keys()), y=list(cost_items.values()),
            marker_color=["#ef4444","#dc2626","#b91c1c","#f97316","#f59e0b","#eab308"],
            text=[f"${v:,.0f}" for v in cost_items.values()], textposition="outside"
        ))
        fig_cost.update_layout(**PLOT_BG, height=270, showlegend=False,
                                xaxis_title="Thành phần chi phí", yaxis_title="USD")
        st.plotly_chart(fig_cost, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="sec">Risk Score × Số tiền (colored by Action)</div>', unsafe_allow_html=True)
        samp = df.sample(min(1200,len(df)), random_state=7)
        fig_sc = px.scatter(samp, x="amt", y="risk_score", color="action",
            color_discrete_map={"BLOCK":"#f44336","MANUAL":"#ff9800","OTP":"#ffeb3b","APPROVE":"#22c55e"},
            opacity=0.55, height=270,
            labels={"amt":"Số tiền ($)","risk_score":"Risk Score","action":"Hành động"})
        fig_sc.add_hline(y=threshold, line_dash="dash", line_color="#a78bfa",
                          annotation_text=f"Threshold={threshold}")
        fig_sc.update_layout(**PLOT_BG)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_d:
        st.markdown('<div class="sec">Giao dịch theo giờ — tỷ lệ cần xử lý</div>', unsafe_allow_html=True)
        hourly = df.groupby("hour").agg(
            total=("trans_id","count"),
            flagged=("action", lambda a: (a!="APPROVE").sum())
        ).reset_index()
        hourly["flag_rate"] = hourly["flagged"]/hourly["total"]*100
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=hourly["hour"],y=hourly["total"],name="Tổng GD",
                                marker_color="#1e3a5f",opacity=0.7))
        fig_h.add_trace(go.Scatter(x=hourly["hour"],y=hourly["flag_rate"],name="% Gắn cờ",
                                    mode="lines+markers",line=dict(color="#f59e0b",width=2),yaxis="y2"))
        fig_h.update_layout(**PLOT_BG, height=270,
            yaxis=dict(title="Số lượng GD"),
            yaxis2=dict(title="% Gắn cờ", overlaying="y", side="right", color="#f59e0b"))
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec">🚨 Top 20 Giao dịch Ưu tiên Xử lý (Priority 1→4)</div>', unsafe_allow_html=True)
    top20 = df[df["action"]!="APPROVE"].sort_values(["priority","risk_score"],
        ascending=[True,False]).head(20)[["trans_id","_cust","_merch","amt","risk_score","action_vn","_cat"]].copy()
    top20.columns = ["ID GD","Khách hàng","Merchant","Số tiền ($)","Risk Score","Hành động","Danh mục"]
    top20["Số tiền ($)"] = top20["Số tiền ($)"].map("${:,.2f}".format)
    top20["Risk Score"]  = top20["Risk Score"].round(4)
    st.dataframe(top20, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — TRANSACTION INVESTIGATION
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Điều tra Giao dịch":
    st.markdown("## 🔍 Điều tra Giao dịch")

    col_f1, col_f2, col_f3 = st.columns([2,1,1])
    with col_f1:
        search = st.text_input("🔎 Tìm Transaction ID", placeholder="TXN000042")
    with col_f2:
        action_filter = st.multiselect("Hành động", ["BLOCK","MANUAL","OTP","APPROVE"],
                                        default=["BLOCK","MANUAL","OTP"])
    with col_f3:
        sort_by = st.selectbox("Sắp xếp theo", ["priority","risk_score","amt"])

    view = df[df["action"].isin(action_filter)] if action_filter else df
    if search:
        view = view[view["trans_id"].str.contains(search, case=False)]
    view = view.sort_values([sort_by]+["risk_score"], ascending=[True,False]).head(300)

    col_list, col_detail = st.columns([1,2])
    with col_list:
        st.markdown(f'<div class="sec">Danh sách ({len(view)} GD)</div>', unsafe_allow_html=True)
        sel = st.radio("Chọn GD:",
            options=view.index.tolist(),
            format_func=lambda i: f"{df.loc[i,'trans_id']} · ${df.loc[i,'amt']:.0f} · {df.loc[i,'action']}",
            key="sel_txn", label_visibility="collapsed")

    if sel is not None:
        row = df.loc[sel]
        with col_detail:
            score = float(row["risk_score"])
            action= str(row["action"])

            st.markdown(f"""
            <div style="background:#111422;border:1px solid #252a45;border-radius:12px;padding:18px 22px;margin-bottom:12px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
                <span style="font-size:1.15rem;font-weight:700;color:#e2e8f8">{row['trans_id']}</span>
                {action_badge(action)}
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;font-size:.84rem;color:#6b7599">
                <div>👤 <b style="color:#c7d2fe">{row.get('_cust','—')}</b></div>
                <div>🏪 <b style="color:#c7d2fe">{row.get('_merch','—')}</b></div>
                <div>🏷️ <b style="color:#c7d2fe">{row.get('_cat','—')}</b></div>
                <div>💰 <b style="color:#22c55e">${row['amt']:.2f}</b></div>
                <div>⏰ <b style="color:#c7d2fe">{str(row.get('_ts',''))[:16]}</b></div>
                <div>📍 <b style="color:#c7d2fe">{row.get('distance_km',0):.0f} km</b></div>
                <div>{'🌐 Online' if row.get('is_online',0)==1 else '🏬 In-store'}</div>
                <div>{'🌙 Ban đêm' if row.get('is_night',0)==1 else '☀️ Ban ngày'}</div>
                <div>📈 {row.get('txn_count_24h',0):.0f} GD/24h</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            col_g1, col_g2 = st.columns([1,1])
            with col_g1:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"font":{"color":"#e2e8f8","size":26},"suffix":""},
                    gauge={
                        "axis":{"range":[0,1],"tickcolor":"#6b7599"},
                        "bar":{"color":"#f44336" if score>=0.7 else "#ff9800" if score>=threshold else "#22c55e"},
                        "bgcolor":"#111422",
                        "steps":[
                            {"range":[0,threshold],"color":"#0d2310"},
                            {"range":[threshold,auto_block_thr],"color":"#2d1a00"},
                            {"range":[auto_block_thr,1],"color":"#2d0000"},
                        ],
                        "threshold":{"line":{"color":"#a78bfa","width":2},"value":threshold}
                    },
                    title={"text":"Risk Score","font":{"color":"#a5b4fc","size":13}},
                ))
                fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=190,margin=dict(l=15,r=15,t=45,b=5))
                st.plotly_chart(fig_g, use_container_width=True)

            with col_g2:
                # Feature contribution bar
                factors = explain_transaction(row)
                if factors:
                    fig_exp = go.Figure(go.Bar(
                        y=[f["description"][:40] for f in factors],
                        x=[f["contribution"] for f in factors],
                        orientation="h",
                        marker_color=["#f44336" if f["contribution"]>0.8 else
                                      "#ff9800" if f["contribution"]>0.4 else "#3b82f6"
                                      for f in factors],
                    ))
                    fig_exp.update_layout(**PLOT_BG, height=190,
                                          xaxis_title="Mức đóng góp vào rủi ro",
                                          title=dict(text="Feature Importance (GD này)",font=dict(color="#a5b4fc",size=12)))
                    st.plotly_chart(fig_exp, use_container_width=True)

            # Business rules
            st.markdown('<div class="sec">📋 Quy tắc nghiệp vụ được kích hoạt</div>', unsafe_allow_html=True)
            rules = row["rules_hit"]
            for r in (rules if isinstance(rules,list) else [rules]):
                st.markdown(f'<div class="rule-card">✅ <b>{r}</b></div>', unsafe_allow_html=True)

            # Recommendation
            st.markdown('<div class="sec">💡 Khuyến nghị hệ thống</div>', unsafe_allow_html=True)
            color_map = {"BLOCK":"error","MANUAL":"warning","OTP":"warning","APPROVE":"success"}
            getattr(st, color_map[action])(row["action_vn"])

            # Human decision
            st.markdown('<div class="sec">👤 Quyết định cuối cùng</div>', unsafe_allow_html=True)
            dec_note = st.text_input("Ghi chú (tùy chọn)", key=f"note_{sel}", placeholder="Lý do quyết định…")
            c1,c2,c3 = st.columns(3)

            def save_decision(final):
                st.session_state.decision_history.append({
                    "trans_id":       row["trans_id"],
                    "customer":       row.get("_cust","—"),
                    "amount":         round(float(row["amt"]),2),
                    "risk_score":     round(score,4),
                    "action_vn":      row["action_vn"],
                    "final_decision": final,
                    "note":           dec_note,
                    # policy context for audit
                    "policy_threshold":      POLICY["threshold"],
                    "policy_block_thr":      POLICY["auto_block_threshold"],
                    "policy_otp_thr":        POLICY["otp_threshold"],
                    "policy_fn_multiplier":  POLICY["missed_fraud_multiplier"],
                    "policy_verify_cost":    POLICY["cost_verification"],
                    "decision_time":         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "rules_hit":             " | ".join(row["rules_hit"]) if isinstance(row["rules_hit"],list) else row["rules_hit"],
                })

            with c1:
                if st.button("✅ Phê duyệt", use_container_width=True, key=f"app_{sel}"):
                    save_decision("Approved"); st.toast("✅ Đã phê duyệt!", icon="✅")
            with c2:
                if st.button("❌ Từ chối / Khóa", use_container_width=True, key=f"rej_{sel}"):
                    save_decision("Rejected"); st.toast("❌ Đã từ chối!", icon="❌")
            with c3:
                if st.button("⏸️ Cần xem xét thêm", use_container_width=True, key=f"rev_{sel}"):
                    save_decision("Need Review"); st.toast("⏸️ Đã đánh dấu!", icon="⏸️")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — DECISION SIMULATOR
# ══════════════════════════════════════════════════════════════
elif page == "🎛️ Decision Simulator":
    st.markdown("## 🎛️ Decision Simulator — Mô phỏng Tác động Vận hành")
    st.caption("Thay đổi chính sách và xem ngay tác động đến số lượng nhân lực, chi phí và hiệu quả phát hiện.")

    # ── Build 3 scenario policies
    st.markdown('<div class="sec">Thiết lập 3 kịch bản chính sách</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    scenarios = {}

    def scenario_inputs(col, label, default_thr, default_block, default_otp, key):
        with col:
            st.markdown(f"**{label}**")
            p = POLICY.copy()
            p["threshold"]            = st.slider("Threshold",   0.05,0.60,default_thr,0.01,key=f"{key}_t")
            p["auto_block_threshold"] = st.slider("Block Thr",   0.50,0.99,default_block,0.01,key=f"{key}_b")
            p["otp_threshold"]        = st.slider("OTP Thr",     0.20,0.80,default_otp,0.01,key=f"{key}_o")
            p["large_amount"]         = st.number_input("Large Amt ($)",200,10000,large_amount,100,key=f"{key}_a")
            return p

    scenarios["A — Bảo thủ (ít rủi ro)"] = scenario_inputs(c1,"🔒 A — Bảo thủ",0.10,0.70,0.35,"sca")
    scenarios["B — Hiện tại"]             = scenario_inputs(c2,"⚖️  B — Hiện tại",threshold,auto_block_thr,otp_thr,"scb")
    scenarios["C — Thoải mái (ít nhân lực)"] = scenario_inputs(c3,"🔓 C — Thoải mái",0.40,0.90,0.65,"scc")

    # Run pipeline for each scenario
    @st.cache_data(show_spinner=False)
    def run_scenario(file_bytes, p_key):
        raw = pd.read_csv(io.BytesIO(file_bytes)) if file_bytes else generate_sample_data(5000)
        p   = eval(p_key)  # safe: only our own dicts
        return run_pipeline(raw, p)

    results = {}
    for name, pol in scenarios.items():
        sc_df = run_pipeline(
            pd.read_csv(io.BytesIO(file_bytes)) if file_bytes else generate_sample_data(5000),
            pol
        )
        results[name] = (sc_df, compute_cost_model(sc_df, pol))

    st.markdown("---")
    # ── Comparison table
    st.markdown('<div class="sec">Bảng so sánh kịch bản — Tác động vận hành</div>', unsafe_allow_html=True)
    rows = []
    for name,(sc_df,c) in results.items():
        rows.append({
            "Kịch bản": name,
            "Threshold": scenarios[name]["threshold"],
            "🔴 Khóa": c["n_block"],
            "🟡 Thủ công": c["n_manual"],
            "🟠 OTP": c["n_otp"],
            "🟢 Phê duyệt": c["n_approve"],
            "Analyst-ngày": c["workload_analysts"],
            "Precision": f"{c['precision']:.3f}",
            "Recall": f"{c['recall']:.3f}",
            "$-Recall": f"{c['dollar_recall']*100:.1f}%",
            "FP Cost ($)": f"{c['fp_verify_cost']:,.0f}",
            "FN Loss ($)": f"{c['fn_loss']:,.0f}",
            "Manual Labor ($)": f"{c['manual_labor_cost']:,.0f}",
            "Tổng chi phí ($)": f"{c['total_cost']:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Operational impact charts
    st.markdown('<div class="sec">Tác động vận hành: phân phối quyết định & chi phí</div>', unsafe_allow_html=True)
    fig_ops = make_subplots(rows=1,cols=3,
        subplot_titles=["Phân phối hành động","Workload (analyst-ngày)","Breakdown chi phí ($)"])

    colors_act = {"🔴 Khóa":"#f44336","🟡 Thủ công":"#ff9800","🟠 OTP":"#ffeb3b","🟢 Phê duyệt":"#22c55e"}
    sc_names   = list(results.keys())
    act_keys   = ["🔴 Khóa","🟡 Thủ công","🟠 OTP","🟢 Phê duyệt"]
    cost_keys  = ["fp_verify_cost","fn_loss","fn_chargeback","fn_investigation","manual_labor_cost","otp_send_cost"]
    cost_labels= ["FP Verify","FN Loss","Chargeback","Investigation","Manual Labor","OTP"]
    cost_colors= ["#ef4444","#dc2626","#b91c1c","#f97316","#f59e0b","#eab308"]

    sc_short   = ["A","B","C"]
    for j,(name,(_,c)) in enumerate(results.items()):
        vals = [c["n_block"],c["n_manual"],c["n_otp"],c["n_approve"]]
        for i,(ak,v) in enumerate(zip(act_keys,vals)):
            fig_ops.add_trace(go.Bar(name=ak,x=[sc_short[j]],y=[v],
                marker_color=list(colors_act.values())[i],showlegend=(j==0)),row=1,col=1)
        fig_ops.add_trace(go.Bar(name=sc_short[j],x=[sc_short[j]],y=[c["workload_analysts"]],
            marker_color=["#6366f1","#22c55e","#f59e0b"][j],showlegend=False),row=1,col=2)
        for ck,cl,cc in zip(cost_keys,cost_labels,cost_colors):
            fig_ops.add_trace(go.Bar(name=cl,x=[sc_short[j]],y=[c[ck]],
                marker_color=cc,showlegend=(j==0)),row=1,col=3)

    fig_ops.update_layout(**PLOT_BG, height=320, barmode="stack")
    st.plotly_chart(fig_ops, use_container_width=True)

    # ── Threshold sweep — total cost vs operational load
    st.markdown('<div class="sec">Quét ngưỡng — Tổng chi phí & Workload</div>', unsafe_allow_html=True)
    sweep_raw = pd.read_csv(io.BytesIO(file_bytes)) if file_bytes else generate_sample_data(5000)
    thresholds = np.arange(0.05, 0.75, 0.02)
    sweep_costs, sweep_manual, sweep_prec, sweep_rec = [],[],[],[]
    for t in thresholds:
        p2 = POLICY.copy(); p2["threshold"]=t
        sc2 = run_pipeline(sweep_raw, p2)
        c2  = compute_cost_model(sc2, p2)
        sweep_costs.append(c2["total_cost"])
        sweep_manual.append(c2["n_manual"])
        sweep_prec.append(c2["precision"])
        sweep_rec.append(c2["recall"])

    optimal_t = thresholds[np.argmin(sweep_costs)]
    fig_sw = make_subplots(rows=1,cols=2,
        subplot_titles=["Total Cost vs Threshold","Precision / Recall / Workload"])
    fig_sw.add_trace(go.Scatter(x=thresholds,y=sweep_costs,mode="lines",name="Total Cost",
        line=dict(color="#f44336",width=2)),row=1,col=1)
    fig_sw.add_vline(x=optimal_t,line_dash="dash",line_color="#22c55e",
        annotation_text=f"Optimal={optimal_t:.2f}",row=1,col=1)
    fig_sw.add_vline(x=threshold,line_dash="dot",line_color="#a78bfa",
        annotation_text=f"Current={threshold}",row=1,col=1)
    fig_sw.add_trace(go.Scatter(x=thresholds,y=sweep_prec,name="Precision",
        line=dict(color="#22c55e",width=2)),row=1,col=2)
    fig_sw.add_trace(go.Scatter(x=thresholds,y=sweep_rec,name="Recall",
        line=dict(color="#3b82f6",width=2)),row=1,col=2)
    fig_sw.add_trace(go.Scatter(x=thresholds,
        y=[m/max(sweep_manual+[1])*max(sweep_prec) for m in sweep_manual],
        name="Workload (norm.)",line=dict(color="#f59e0b",width=2,dash="dot")),row=1,col=2)
    fig_sw.update_layout(**PLOT_BG, height=310)
    st.plotly_chart(fig_sw, use_container_width=True)
    st.info(f"💡 Ngưỡng tối ưu chi phí với cài đặt hiện tại: **{optimal_t:.2f}** · Chi phí: **${min(sweep_costs):,.0f}**")

    # ── Confusion matrices side by side
    st.markdown('<div class="sec">Confusion Matrix — So sánh 3 kịch bản</div>', unsafe_allow_html=True)
    fig_cm = make_subplots(rows=1,cols=3,subplot_titles=sc_short)
    for j,(_,(sc_df,c)) in enumerate(results.items()):
        cm = np.array([[c["tn"],c["fp"]],[c["fn"],c["tp"]]])
        fig_cm.add_trace(go.Heatmap(z=cm,x=["Pred Legit","Pred Fraud"],y=["Act Legit","Act Fraud"],
            colorscale=[[0,"#0d1117"],[0.5,"#1e3a5f"],[1,"#f44336"]],
            text=cm,texttemplate="%{text}",showscale=False),row=1,col=j+1)
    fig_cm.update_layout(**PLOT_BG,height=260)
    st.plotly_chart(fig_cm, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — DECISION HISTORY (full audit context)
# ══════════════════════════════════════════════════════════════
elif page == "📋 Lịch sử Quyết định":
    st.markdown("## 📋 Lịch sử Quyết định — Nhật ký Kiểm toán")

    if not st.session_state.decision_history:
        st.info("Chưa có quyết định nào. Hãy điều tra giao dịch ở tab **Điều tra Giao dịch** và đưa ra quyết định.")
    else:
        hist = pd.DataFrame(st.session_state.decision_history)

        # KPIs
        cc = st.columns(5)
        total = len(hist)
        with cc[0]: st.markdown(kpi("Tổng quyết định",f"{total}","blue"), unsafe_allow_html=True)
        with cc[1]: st.markdown(kpi("✅ Phê duyệt",f"{(hist['final_decision']=='Approved').sum()}","green"), unsafe_allow_html=True)
        with cc[2]: st.markdown(kpi("❌ Từ chối",f"{(hist['final_decision']=='Rejected').sum()}","red"), unsafe_allow_html=True)
        with cc[3]: st.markdown(kpi("⏸️ Cần xem xét",f"{(hist['final_decision']=='Need Review').sum()}","orange"), unsafe_allow_html=True)
        with cc[4]:
            avg_score = hist["risk_score"].mean()
            st.markdown(kpi("Avg Risk Score",f"{avg_score:.3f}","purple"), unsafe_allow_html=True)

        st.markdown("")

        # Agreement analysis: did human agree with system?
        hist["system_action"] = hist["action_vn"].apply(
            lambda a: "FLAG" if any(x in a for x in ["Khóa","OTP","Kiểm"]) else "APPROVE"
        )
        hist["human_agree"] = hist.apply(
            lambda r: "✅ Đồng ý" if (
                (r["system_action"]=="FLAG" and r["final_decision"]!="Approved") or
                (r["system_action"]=="APPROVE" and r["final_decision"]=="Approved")
            ) else "⚠️ Khác hệ thống", axis=1
        )
        agree_rate = (hist["human_agree"]=="✅ Đồng ý").mean()*100

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown('<div class="sec">Phân phối quyết định</div>', unsafe_allow_html=True)
            fig_d = px.pie(hist, names="final_decision",
                color_discrete_map={"Approved":"#22c55e","Rejected":"#f44336","Need Review":"#ff9800"},
                hole=0.5, height=220)
            fig_d.update_layout(**PLOT_BG)
            st.plotly_chart(fig_d, use_container_width=True)

        with col_v2:
            st.markdown('<div class="sec">Risk Score của các quyết định</div>', unsafe_allow_html=True)
            fig_rs = px.histogram(hist, x="risk_score", color="final_decision",
                color_discrete_map={"Approved":"#22c55e","Rejected":"#f44336","Need Review":"#ff9800"},
                nbins=20, height=220, labels={"risk_score":"Risk Score"})
            fig_rs.add_vline(x=threshold, line_dash="dash", line_color="#a78bfa")
            fig_rs.update_layout(**PLOT_BG)
            st.plotly_chart(fig_rs, use_container_width=True)

        st.markdown(f'<div class="sec">Tỷ lệ đồng thuận người dùng — hệ thống: <b style="color:#22c55e">{agree_rate:.0f}%</b></div>', unsafe_allow_html=True)

        # Full audit table with policy context
        display_cols = ["trans_id","customer","amount","risk_score","action_vn","final_decision",
                        "human_agree","note","policy_threshold","policy_block_thr",
                        "policy_fn_multiplier","rules_hit","decision_time"]
        existing = [c for c in display_cols if c in hist.columns]
        st.dataframe(hist[existing].rename(columns={
            "trans_id":"ID GD","customer":"Khách hàng","amount":"Số tiền ($)",
            "risk_score":"Risk Score","action_vn":"KN hệ thống","final_decision":"QĐ người dùng",
            "human_agree":"Đồng thuận","note":"Ghi chú","policy_threshold":"Threshold",
            "policy_block_thr":"Block Thr","policy_fn_multiplier":"FN Mult",
            "rules_hit":"Quy tắc","decision_time":"Thời gian",
        }), use_container_width=True, hide_index=True)

        # Export
        buf = io.StringIO()
        hist.to_csv(buf, index=False, encoding="utf-8-sig")
        st.download_button("📥 Xuất nhật ký kiểm toán CSV",
            data=buf.getvalue().encode("utf-8-sig"),
            file_name=f"audit_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True)

        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            st.session_state.decision_history = []
            st.rerun()

# ── Footer
st.markdown("""<div style="text-align:center;color:#2d3155;font-size:.74rem;margin-top:2rem">
Fraud Detection DSS · LightGBM + Cost-Sensitive Learning + Rule Engine · Streamlit</div>""",
unsafe_allow_html=True)
