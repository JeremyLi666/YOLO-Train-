import pandas as pd
from datetime import timedelta
import ccxt

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50000)


p = {'http': '127.0.0.1:7897', 'https': '127.0.0.1:7897'}
exchange = ccxt.binance({'proxies': p})


symbol = 'BTC/USDT'
time_interval = '1h'


data = exchange.fetch_ohlcv(symbol=symbol, timeframe=time_interval)

df = pd.DataFrame(data, dtype=float)
df.rename(columns={0: 'MTS', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['MTS'], unit='ms') + timedelta(hours=8)  # 转换为北京时间，命名为date

df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
df['date'] = df['date'].dt.strftime('%Y-%m-%d')
df.to_csv('#data_BTC_1h.csv', index=False)
