import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
# 2. 한글 폰트 설정 (Windows)
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

# 24:00:00 제거 (비정상 시간)
df = df[df["시분초"] != "24:00:00"]

# datetime 생성
df["datetime"] = pd.to_datetime(
    df["연월일"] + " " + df["시분초"]
)

# 시간 파생 변수
df["월"] = df["datetime"].dt.month
df["시간"] = df["datetime"].dt.hour
df["요일"] = df["datetime"].dt.dayofweek

# =========================
# 5. 태양광 특성 반영
# =========================
# 태양광 발전이 없는 야간 시간대 제거 (가장 중요)
df = df[df["시간"].between(6, 18)]

# 필요한 컬럼만 사용
df = df[
    ["발전량", "용량_메가", "월", "시간", "요일"]
]

# 발전량 단위 보정 (Wh → kWh로 가정)
df["발전량"] = df["발전량"] / 1000

# =========================
# 6. 학습 데이터 구성
# =========================
X = df.drop("발전량", axis=1)
y = df["발전량"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 7. 모델 학습
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

# =========================
# 8. 예측
# =========================
y_pred = model.predict(X_test)
y_test_reset = y_test.reset_index(drop=True)
y_pred_reset = pd.Series(y_pred)

# =========================
# 9. 평가 지표
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("※ 주간 시간대(06~18시) 데이터 기준 평가")
print(f"평균 절대 오차(MAE): {mae:,.2f} kWh")
print(f"평균 제곱근 오차(RMSE): {rmse:,.2f} kWh")
print(f"결정계수(R²): {r2:.3f}")

# =========================
# 10. 시각화 ① 실제 vs 예측 발전량
# =========================

# 시각화용 샘플링 (점 과밀 방지)
sample_idx = y_test_reset.sample(
    min(3000, len(y_test_reset)),
    random_state=42
).index

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_reset.loc[sample_idx],
    y_pred_reset.loc[sample_idx],
    alpha=0.5,
    color="tab:blue",
    label="예측 발전량"
)

max_val = max(y_test_reset.max(), y_pred_reset.max())
plt.plot(
    [0, max_val],
    [0, max_val],
    color="red",
    linestyle="--",
    linewidth=2,
    label="완벽한 예측 기준선 (y = x)"
)

plt.xlabel("실제 발전량 (kWh)")
plt.ylabel("예측 발전량 (kWh)")
plt.title("실제 발전량 vs 예측 발전량 비교 (주간 데이터)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 11. 시각화 ② 시간대별 평균 발전량
# =========================
hourly_avg = df.groupby("시간")["발전량"].mean()

plt.figure(figsize=(8, 5))
plt.plot(
    hourly_avg.index,
    hourly_avg.values,
    marker="o"
)

plt.xlabel("시간")
plt.ylabel("평균 발전량 (kWh)")
plt.title("시간대별 평균 태양광 발전량")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
