import pandas as pd
import ccxt
import time
import os
from datetime import datetime, timedelta

# 显示设置
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50000)

# 设置代理（如不使用请注释）
proxies = {'http': '127.0.0.1:XXXX', 'https': '127.0.0.1:XXXX'}

# 初始化交易所
exchange = ccxt.binance({
    'enableRateLimit': True,
    'proxies': proxies,
})

# 参数设置
symbol = 'BTC/USDT'
timeframe = '1m'
limit = 1000

start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 7, 1)
output_file = 'Data_BTC_m.csv'

# 是否写入 header（只在首次写入时）
write_header = not os.path.exists(output_file)

# 遍历日期
current_date = start_date
while current_date < end_date:
    print(f"\n日期：{current_date.strftime('%Y-%m-%d')}")

    # 每天两段（00:00–12:00，12:00–24:00）
    segments = [
        (current_date, current_date + timedelta(hours=12)),
        (current_date + timedelta(hours=12), current_date + timedelta(days=1))
    ]

    for idx, (start_time, end_time) in enumerate(segments):
        since = int(start_time.timestamp() * 1000)
        until = int(end_time.timestamp() * 1000)

        print(f"  段 {idx+1}: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
        try:
            ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"    -> 报错: {str(e)}，重试中...")
            time.sleep(5)
            continue

        # 过滤掉超出时间范围的数据
        ohlcv = [row for row in ohlcv if since <= row[0] < until]

        if not ohlcv:
            print("    -> 无数据，跳过此段")
            continue

        # 构造 DataFrame 并格式化时间
        df = pd.DataFrame(ohlcv, columns=['MTS', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['MTS'], unit='ms') + timedelta(hours=8)  # UTC 转北京时间
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')

        # 写入文件（第一轮写 header，其余不写）
        df.to_csv(output_file, mode='a', header=write_header, index=False)
        write_header = False  # 后续都不写表头

        print(f"    -> 写入成功，共 {len(df)} 条")

        time.sleep(0.4)  # 避免被限频

    current_date += timedelta(days=1)
