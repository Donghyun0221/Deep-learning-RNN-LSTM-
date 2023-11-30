import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 모델 불러오기
model = load_model('qkrdata10.model')

# 데이터 불러오기 및 스케일링
df = pd.read_excel('qkrdata10.xlsx', engine='openpyxl')
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# 최근 48시간 데이터 추출
recent_data = df_scaled[-144:]  # 최근 144개의 데이터
recent_data = np.reshape(recent_data, (1, recent_data.shape[0], 1))

# 12시간 예측 수행
predicted = model.predict(recent_data)

# 예측 결과 스케일 역변환
predicted_inverse = scaler.inverse_transform(predicted)

start_time = pd.to_datetime('2023-10-27 08:00', format='%Y-%m-%d %H:%M')


time_periods = [start_time + pd.DateOffset(minutes=10 * i) for i in range(1, 13)]


predicted_inverse = predicted_inverse.flatten()[:len(time_periods)]

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(time_periods, predicted_inverse, label='Predicted')
plt.title('qkr 12-Hour Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')

# x축 눈금을 시간 형식으로 설정
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))  # 10분 간격으로 눈금 표시

plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)

# 예측 결과 출력
print("Predicted Prices for the Next 12 Hours:")
for i, (time, price) in enumerate(zip(time_periods, predicted_inverse)):
    print(f"Time {i+1}: {time}, Price: {price:.2f}")

# 결과를 데이터프레임으로 변환
predicted_data = {'Date': time_periods, 'Close': predicted_inverse}
predicted_df = pd.DataFrame(predicted_data)

# 엑셀 파일로 저장
predicted_df.to_excel('predicted10_prices.xlsx', index=False)

# 저장 완료 메시지 출력
print("Predicted prices have been saved to 'predicted_prices.xlsx'")

# 그래프 출력
plt.show()