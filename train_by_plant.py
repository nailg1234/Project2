import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# =========================
# 1. CSV 안전 로드 함수
# =========================
def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8", index_col=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", index_col=False)

# =========================
# 2. 한글 폰트 설정
# =========================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 3. 데이터 로드
# =========================
df = read_csv_safe(
    "data/한국서부발전(주)_(AI친화)신재생 에너지 발전량 데이터_20221231.csv"
)

# =========================
# 4. 데이터 정제
# =========================
df = df[df["시분초"] != "24:00:00"]

df["datetime"] = pd.to_datetime(
    df["연월일"] + " " + df["시분초"]
)

df["월"] = df["datetime"].dt.month
df["시간"] = df["datetime"].dt.hour
df["요일"] = df["datetime"].dt.dayofweek

# 주간 시간대
df = df[df["시간"].between(6, 18)]

# 발전량 단위 보정 (Wh → kWh 가정)
df["발전량"] = df["발전량"] / 1000

# =========================
# 5. 발전소 목록
# =========================
plants = df["발전기명"].unique()
print(f"총 발전소 수: {len(plants)}")

# =========================
# 6. 발전소별 학습 & 시각화
# =========================
for plant in plants:
    print(f"\n===== 발전소: {plant} =====")

    df_p = df[df["발전기명"] == plant].copy()

    if len(df_p) < 300:
        print("데이터 수 부족으로 스킵")
        continue

    # =========================
    # Feature / Target
    # =========================
    X = df_p[["용량_메가", "월", "시간", "요일"]]
    y = df_p["발전량"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # 모델
    # =========================
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # =========================
    # 평가
    # =========================
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:,.2f} kWh")
    print(f"RMSE: {rmse:,.2f} kWh")
    print(f"R²: {r2:.3f}")

    # =========================
    # 시각화 ① 산점도 (유지)
    # =========================
    y_test_r = y_test.reset_index(drop=True)
    y_pred_r = pd.Series(y_pred)

    sample_idx = y_test_r.sample(
        min(1500, len(y_test_r)),
        random_state=42
    ).index

    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_test_r.loc[sample_idx],
        y_pred_r.loc[sample_idx],
        alpha=0.5,
        label="예측값"
    )

    max_val = max(y_test_r.max(), y_pred_r.max())
    plt.plot(
        [0, max_val],
        [0, max_val],
        "r--",
        linewidth=2,
        label="이상적 예측"
    )

    plt.xlabel("실제 발전량 (kWh)")
    plt.ylabel("예측 발전량 (kWh)")
    plt.title(f"{plant} | 실제 vs 예측")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # =========================
    # ★ 시각화 ② 발전소별 시간대 평균 발전량
    # =========================
    hourly_avg = df_p.groupby("시간")["발전량"].mean()

    plt.figure(figsize=(7, 4))
    plt.plot(
        hourly_avg.index,
        hourly_avg.values,
        marker="o"
    )

    plt.xlabel("시간")
    plt.ylabel("평균 발전량 (kWh)")
    plt.title(f"{plant} | 시간대별 평균 발전량")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
