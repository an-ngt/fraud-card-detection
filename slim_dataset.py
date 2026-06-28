import argparse
from pathlib import Path

import pandas as pd

# ── Cột mà app.py cần để chạy đầy đủ ──────────────────────────────────────
# Các cột được giữ lại và lý do:
KEEP_COLS = {
    # Định danh khách hàng — dùng để tính behavioral features (cc_num groupby)
    "cc_num",
    # Thời gian — tính trans_hour → sin/cos/is_night; unix_time → time_since_last_txn, txn_count_24h
    "trans_date_trans_time",
    "unix_time",
    # Ngày sinh — tính tuổi (age)
    "dob",
    # Số tiền — amt_log, amt_ratio, cost function
    "amt",
    # Danh mục — category_risk_tier, is_online
    "category",
    # Cửa hàng — merchant_freq
    "merchant",
    # Giới tính — feature trực tiếp
    "gender",
    # Tọa độ — Haversine → distance_km
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    # Nhãn (tùy chọn — bỏ đi nếu muốn test không nhãn)
    "is_fraud",
}

# Cột bổ sung có thể hữu ích cho hiển thị (không bắt buộc cho model)
OPTIONAL_DISPLAY_COLS = {
    "city",
    "state",
    "city_pop",
}


def slim(input_path: str, output_path: str, keep_display: bool = True) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"[1/3] Đọc file gốc: {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    original_size_mb = input_path.stat().st_size / 1_048_576
    print(f"      → {len(df):,} dòng × {df.shape[1]} cột  ({original_size_mb:.0f} MB)")

    cols_to_keep = KEEP_COLS.copy()
    if keep_display:
        cols_to_keep |= OPTIONAL_DISPLAY_COLS

    available = [c for c in cols_to_keep if c in df.columns]
    dropped = [c for c in df.columns if c not in cols_to_keep]

    print(f"[2/3] Giữ lại {len(available)} cột, bỏ {len(dropped)} cột:")
    print(f"      Bỏ: {dropped}")

    df_slim = df[available].copy()

    # Ép kiểu để tiết kiệm bộ nhớ trước khi ghi
    if "cc_num" in df_slim.columns:
        df_slim["cc_num"] = df_slim["cc_num"].astype(str)
    for float_col in ["lat", "long", "merch_lat", "merch_long", "amt"]:
        if float_col in df_slim.columns:
            df_slim[float_col] = pd.to_numeric(df_slim[float_col], errors="coerce").astype("float32")
    if "is_fraud" in df_slim.columns:
        df_slim["is_fraud"] = df_slim["is_fraud"].astype("int8")

    print(f"[3/3] Ghi file slim: {output_path} ...")
    df_slim.to_csv(output_path, index=False)

    slim_size_mb = output_path.stat().st_size / 1_048_576
    reduction = (1 - slim_size_mb / original_size_mb) * 100
    print(f"      ✅ Xong! {slim_size_mb:.0f} MB (giảm {reduction:.0f}% so với bản gốc)")

    if slim_size_mb > 200:
        print(
            f"\n  ⚠️  File vẫn còn {slim_size_mb:.0f} MB > 200 MB.\n"
            "      Thêm file .streamlit/config.toml vào repo để tăng giới hạn lên 1 GB\n"
            "      (xem hướng dẫn đi kèm), hoặc chạy lại với --no-display để bỏ cột hiển thị."
        )
    else:
        print(f"\n  👍  File {slim_size_mb:.0f} MB — trong giới hạn 200 MB mặc định của Streamlit Cloud.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thu gọn bộ dữ liệu cho Streamlit Cloud.")
    parser.add_argument("--input", default="transactions.csv", help="Đường dẫn file CSV gốc")
    parser.add_argument("--output", default="transactions_slim.csv", help="Đường dẫn file CSV đầu ra")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Bỏ cả các cột hiển thị phụ (city, state, city_pop) để file nhỏ hơn nữa",
    )
    args = parser.parse_args()
    slim(args.input, args.output, keep_display=not args.no_display)
