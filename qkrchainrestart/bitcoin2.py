import pyupbit
import pandas as pd

# 원하는 티커 목록을 작성합니다. 이더리움 (ETH)과 쿼크체인 (QKC)을 추가합니다.
tickers = ["KRW-QKC"]

# 현재 가격을 가져옵니다.
current_prices = pyupbit.get_current_price(tickers)
print(current_prices)

# 이후의 코드를 실행할 수 있습니다.

ticker = 'KRW-QKC'
interval = 'minute60'
to = '2023-10-26 18:00'
count = 4000

# Adjusted the 'count' value to a reasonable number. 10000 is too large and may not return data.

ohlcv_data4000 = pyupbit.get_ohlcv(ticker=ticker, interval=interval, to=to, count=count)
print(ohlcv_data4000)

# Save the historical data to an Excel file.
ohlcv_data4000.to_excel('ohlcv_data4000.xlsx', engine='openpyxl')