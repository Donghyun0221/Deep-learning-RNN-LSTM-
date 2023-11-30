import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 불러오기 및 스케일링
df = pd.read_excel('qkrdataday.xlsx', engine='openpyxl')
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# 데이터 준비
seq_len = 15
future_period = 8
data_cnt = len(df['Close'])
result = []

for idx in range(data_cnt - seq_len - future_period):
    seq_x = df['Close'][idx: idx + seq_len]
    seq_y = df['Close'][idx + seq_len: idx + seq_len + future_period]
    result.append(np.append(seq_x, seq_y))

result = np.array(result)
row_cnt = int(round(result.shape[0] * 0.8))

# 훈련 및 테스트 데이터 분할
train_data = result[:row_cnt, :]
x_train = train_data[:, :seq_len]
y_train = train_data[:, seq_len:]
x_train_reshape = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = result[row_cnt:, :seq_len]
x_test_reshape = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
y_test = result[row_cnt:, seq_len:]

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(15, return_sequences=True, input_shape=(15, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(future_period, activation='linear'))
model.compile(loss='mse', optimizer='adam')

# 모델 훈련
model.fit(x_train_reshape, y_train, validation_data=(x_test_reshape, y_test), batch_size=32, epochs=20)

# 모델 저장
model.save('qkrdataDay.model')

# 예측
pred = model.predict(x_test_reshape)

# 예측값 역 스케일링
y_test_original = scaler.inverse_transform(y_test)
pred_original = scaler.inverse_transform(pred)

# 평가
mse = mean_squared_error(y_test_original, pred_original)
r2 = r2_score(y_test_original, pred_original)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# 시각화
plt.figure(figsize=(20, 10))
plt.plot(y_test_original, label='True')
plt.plot(pred_original, label='Prediction')
plt.title(f"day")
plt.legend()
plt.tight_layout()
plt.show()