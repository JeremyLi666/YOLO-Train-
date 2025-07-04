import pandas as pd
import ccxt
import time
from datetime import datetime, timedelta

# 设置显示选项
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50000)

# 设置代理（如不使用代理可注释掉）
proxies = {'http': '127.0.0.1:你的端口', 'https': '127.0.0.1:你的端口'}

# 初始化 Binance
exchange = ccxt.binance({
    'enableRateLimit': True,
    'proxies': proxies,
})

# 参数配置
symbol = 'BTC/USDT'
timeframe = '1d'
limit = 1500  # Binance 最大支持 1500
since = exchange.parse8601('xxxx-xx-xxT00:00:00Z')  # 从哪一天开始抓取
all_data = []

# 主循环：持续拉取，直到数据不再增加
while True:
    print(f'Fetching since {exchange.iso8601(since)}...')
    try:
        ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)
    except Exception as e:
        print('出错:', str(e))
        time.sleep(5)
        continue

    if not ohlcv:
        break

    all_data += ohlcv

    # 判断是否拉到最后一段数据
    if len(ohlcv) < limit:
        break

    # 更新 since 为最后一条数据的时间 + 1ms
    since = ohlcv[-1][0] + 1
    time.sleep(0.5)

# 转为 DataFrame
df = pd.DataFrame(all_data, columns=['MTS', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = pd.to_datetime(df['MTS'], unit='ms') + timedelta(hours=8)  # UTC转北京时间
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# 保存
df.to_csv('#data_2_BTC_1d.csv', index=False)
print(f'完成，共获取 {len(df)} 条数据，保存为 #Raw_data.csv')

