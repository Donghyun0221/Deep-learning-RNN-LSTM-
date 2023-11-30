import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 모델 불러오기
model = load_model('qkrdata60.model')

# 데이터 불러오기 및 스케일링
df = pd.read_excel('qkrdata60.xlsx', engine='openpyxl')
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# 최근 48시간 데이터 추출
recent_data = df_scaled[-48:] # 최근 48시간 데이터
recent_data = np.reshape(recent_data, (1, recent_data.shape[0], 1))

# 12시간 예측 수행
predicted = model.predict(recent_data)

# 예측 결과 스케일 역변환
predicted_inverse = scaler.inverse_transform(predicted)

# 시간 설정 (예: 2023-10-27-08-00부터 1시간 간격으로)
start_time = pd.to_datetime('2023-10-27 08:00')
time_periods = [start_time + pd.DateOffset(hours=i) for i in range(1, 13)]

# 시간 형식 포맷 지정
formatted_time_periods = [time.strftime('%Y-%m-%d %H:%M') for time in time_periods]

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(formatted_time_periods, predicted_inverse.flatten(), label='Predicted')
plt.title('qkr 12-Hour Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close')

# x축 눈금을 시간 형식으로 설정 (1시간 간격)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1)) # 1시간 간격으로 눈금 표시

plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# 예측 결과 출력 및 저장
print("Predicted Prices for the Next 12 Hours:")
predicted_data = [(formatted_time_periods[i], predicted_inverse[0][i]) for i in range(len(formatted_time_periods))]

# 데이터를 DataFrame으로 만들어 엑셀 파일로 저장
prediction_df = pd.DataFrame(predicted_data, columns=['Date', 'Close'])
prediction_df.to_excel('predicted_prices.xlsx', index=False)