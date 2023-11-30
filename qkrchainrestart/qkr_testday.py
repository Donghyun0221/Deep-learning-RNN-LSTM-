import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 모델 불러오기
model = load_model('qkrdataDay.model')

# 데이터 불러오기 및 스케일링
df = pd.read_excel('qkrdataday.xlsx', engine='openpyxl')
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# 최근 데이터 추출 (예: 마지막 15일)
recent_data = df_scaled[-15:]  # 최근 15일 데이터
recent_data = np.reshape(recent_data, (1, recent_data.shape[0], 1))

# 예측 수행
predicted = model.predict(recent_data)

# 예측 결과 스케일 역변환
predicted_inverse = scaler.inverse_transform(predicted)

# 시간 설정 (예: 2023-10-27 08:00부터 1일 간격으로)
start_time = pd.to_datetime('2023-10-27 08:00', format='%Y-%m-%d %H:%M')
time_periods = [start_time + pd.DateOffset(days=i) for i in range(8)]  # 최근 8일간 예측

# 예측 결과 시각화 (처음 8개 예측만 사용)
plt.figure(figsize=(12, 6))
plt.plot(time_periods, predicted_inverse[0][:8], label='Predicted')  # 1차원 배열로 변경하여 사용
plt.title('qkr Daily Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')

# x축 눈금을 날짜 형식으로 설정
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1일 간격으로 눈금 표시

plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)

# 예측 결과 출력
print("Predicted Prices for the Last 8 Days:")
for i, (date, price) in enumerate(zip(time_periods, predicted_inverse[0][:8])):
    print(f"Date {i+1}: {date.strftime('%Y-%m-%d')}, Price: {price:.2f}")

# 결과를 데이터프레임으로 변환
predicted_data = {'Date': time_periods, 'Price': predicted_inverse[0][:8]}
predicted_df = pd.DataFrame(predicted_data)

# 엑셀 파일로 저장
predicted_df.to_excel('predictedday_prices.xlsx', index=False)

# 저장 완료 메시지 출력
print("Predicted prices have been saved to 'predicted_prices.xlsx'")

# 그래프 출력
plt.show()