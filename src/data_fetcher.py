"""
数据获取模块：通过 akshare 获取A股指数数据和隔夜因子数据。
"""
import time
import pandas as pd
import akshare as ak

# A股主要指数代码映射
INDEX_CODES = {
    "沪深300": "000300",
    "中证500": "000905",
    "中证1000": "000852",
}

# 隔夜因子：海外市场指数
OVERSEAS_INDICES = {
    "标普500": "SPX",
    "纳斯达克": "NDX",
    "道琼斯": "DJI",
}

# 港股现货指数（注意：akshare stock_hk_index_daily_sina 返回的是现货指数，非期货）
# 现货指数无夜盘，open = 早盘开盘(09:15 HKT)，无歧义
HK_INDICES = {
    "恒生指数": "HSI",         # 恒生指数（现货），09:15-16:00
    "恒生科技指数": "HSTECH",  # 恒生科技指数（现货）
    "国企指数": "HSCEI",       # H股指数（现货）
}

# 富时A50期货（新加坡交易所，A股最直接的隔夜参考，夜盘至次日04:45）
A50_SYMBOL = "A50"

# 美国上市的中国相关ETF（美股时段交易，反映海外资金对A股的定价）
US_CHINA_ETFS = {
    "FXI": "FXI",    # iShares 中国大盘ETF（追踪大型中概股/H股）
    "KWEB": "KWEB",  # KraneShares 中国互联网ETF
    "ASHR": "ASHR",  # Xtrackers 沪深300 ETF（直接挂钩A股）
    "MCHI": "MCHI",  # iShares MSCI中国ETF
}

# 大宗商品期货
COMMODITY_FUTURES = {
    "WTI原油": "CL",
    "布伦特原油": "OIL",
    "COMEX铜": "HG",
}

# 欧洲指数（收盘于北京时间次日 ~00:30-01:00，A股开盘前可用）
EURO_INDICES = {
    "欧洲斯托克50": "STOXX50",  # Euro Stoxx 50
    "德国DAX": "DAX",
    "英国富时100": "FTSE",
}

# VIX 恐慌指数
VIX_SYMBOL = "VIX"


def fetch_index_daily(code: str, name: str, start_date: str = "20180101") -> pd.DataFrame:
    """获取A股指数日线数据。"""
    print(f"  获取 {name} ({code}) 日线数据 ...")
    df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start_date)
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_us_index_daily(symbol: str, name: str) -> pd.DataFrame:
    """获取美股指数日线数据（作为隔夜因子）。"""
    print(f"  获取 {name} ({symbol}) 数据 ...")
    try:
        df = ak.index_us_stock_sina(symbol=f".{symbol}")
        df = df.rename(columns={"date": "date", "close": f"us_{symbol}_close", "open": f"us_{symbol}_open"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        # 只保留日期和收盘价、涨跌
        keep_cols = ["date", f"us_{symbol}_close", f"us_{symbol}_open"]
        df = df[[c for c in keep_cols if c in df.columns]]
        return df
    except Exception as e:
        print(f"  [警告] 获取 {name} 数据失败: {e}")
        return pd.DataFrame()


def fetch_gold_price() -> pd.DataFrame:
    """获取黄金价格数据。"""
    print("  获取黄金价格数据 ...")
    try:
        df = ak.spot_golden_benchmark_sge(
            start_date="20180101",
            end_date=pd.Timestamp.now().strftime("%Y%m%d"),
        )
        df = df.rename(columns={"日期": "date", "价格": "gold_price"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "gold_price"]].sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取黄金数据失败: {e}")
        return pd.DataFrame()


def fetch_usd_cny() -> pd.DataFrame:
    """获取美元兑人民币汇率。"""
    print("  获取美元兑人民币汇率 ...")
    try:
        df = ak.fx_spot_quote()
        usd_row = df[df["货币对"].str.contains("USD/CNY", na=False)]
        if not usd_row.empty:
            print("  [信息] 获取到实时汇率快照")
        # 使用央行汇率中间价作为替代
        df = ak.currency_boc_safe(
            symbol="美元",
            start_date="20180101",
            end_date=pd.Timestamp.now().strftime("%Y%m%d"),
        )
        df = df.rename(columns={"日期": "date", "中行汇买价": "usd_cny"})
        df["date"] = pd.to_datetime(df["date"])
        df["usd_cny"] = pd.to_numeric(df["usd_cny"], errors="coerce")
        df = df[["date", "usd_cny"]].dropna().sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取汇率数据失败: {e}")
        return pd.DataFrame()


def fetch_hk_index_daily(symbol: str, name: str) -> pd.DataFrame:
    """获取港股现货指数日线数据。

    注意：akshare stock_hk_index_daily_sina 返回的是现货指数（非期货）。
    现货指数无夜盘交易，因此：
    - open = 早盘开盘价 (09:15 HKT = 09:15 BJT)，无歧义
    - close = 午盘收盘价 (16:00 HKT = 16:00 BJT)
    open 在 A股开盘(09:30)前 15 分钟已知，可直接用于预测。
    """
    print(f"  获取 {name} 现货指数 ({symbol}) 数据 ...")
    try:
        df = ak.stock_hk_index_daily_sina(symbol=symbol)
        df = df.rename(columns={
            "date": "date",
            "close": f"hk_{symbol}_close",
            "open": f"hk_{symbol}_open",
        })
        df["date"] = pd.to_datetime(df["date"])
        keep_cols = ["date", f"hk_{symbol}_close", f"hk_{symbol}_open"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取 {name} 数据失败: {e}")
        return pd.DataFrame()


def fetch_us_china_etf(symbol: str, name: str) -> pd.DataFrame:
    """获取美国上市的中国相关ETF日线数据。

    这些ETF在美股时段交易（北京时间21:30-04:00），反映了海外资金对A股/中概股的定价，
    是A股次日开盘的重要参考。其中ASHR直接追踪沪深300。
    """
    print(f"  获取美股中国ETF {name} ({symbol}) 数据 ...")
    try:
        df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
        df = df.rename(columns={
            "date": "date",
            "close": f"etf_{symbol}_close",
            "open": f"etf_{symbol}_open",
        })
        df["date"] = pd.to_datetime(df["date"])
        keep_cols = ["date", f"etf_{symbol}_close", f"etf_{symbol}_open"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取 {name} 数据失败: {e}")
        return pd.DataFrame()


def _detect_open_convention(df: pd.DataFrame, open_col: str, close_col: str) -> str:
    """经验检测期货日线 open 代表日盘开盘还是夜盘开盘。

    原理：
    - 如果 open = 日盘开盘(09:00)，open[d] 与 close[d-1](前日16:30) 之间隔了一整夜，
      |open[d] - close[d-1]| / close[d-1] 的中位数通常 > 0.3%
    - 如果 open = 夜盘开盘(前夜17:00)，与 close[d-1](同日16:30) 只隔30分钟，
      |open[d] - close[d-1]| / close[d-1] 的中位数通常 < 0.1%
    """
    if open_col not in df.columns or close_col not in df.columns:
        return "unknown"
    gap = (df[open_col] / df[close_col].shift(1) - 1).abs() * 100
    median_gap = gap.median()
    if median_gap < 0.1:
        return "night_session_open"
    elif median_gap > 0.3:
        return "day_session_open"
    else:
        return "ambiguous"


def fetch_a50_futures() -> pd.DataFrame:
    """获取富时中国A50期货数据（新加坡交易所）。

    ===== 重要：open 价格的含义 =====
    SGX 的交易日定义：
    - T+1 session（夜盘）: 前日 17:00 → 当日 04:45
    - T session（日盘）:   当日 09:00 → 当日 16:30

    akshare 的 futures_foreign_hist 返回的日线 open 大概率是夜盘开盘价（前日17:00），
    而非日盘开盘价（当日09:00）。这意味着 open[d+1]/close[d] ≈ 0%，无法衡量隔夜涨跌。

    解决策略（features.py 中实现）：
    1. 自动检测 open 的含义（经验分析 open-close 的价差分布）
    2. 如果 open = 夜盘开盘 → night_ret 会接近 0，模型自动忽略
    3. 依赖 close-to-close return（a50_ret）和其他产品（ASHR、HK现货）获取隔夜信号

    增强策略（尝试获取日盘真实开盘价）：
    1. yfinance 小时线数据 → 提取 09:00 bar 的开盘价
    2. 如成功，替换 a50_open 为真实的日盘开盘价
    """
    print("  获取富时A50期货数据 ...")

    # 策略1: 基础日线数据 (akshare)
    a50_daily = pd.DataFrame()
    try:
        a50_daily = ak.futures_foreign_hist(symbol="A50")
        a50_daily = a50_daily.rename(columns={
            "date": "date", "日期": "date",
            "收盘价": "a50_close", "close": "a50_close",
            "开盘价": "a50_open", "open": "a50_open",
        })
        a50_daily["date"] = pd.to_datetime(a50_daily["date"])
        keep_cols = ["date", "a50_close", "a50_open"]
        a50_daily = a50_daily[[c for c in keep_cols if c in a50_daily.columns]]
        a50_daily = a50_daily.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  [警告] akshare A50日线数据获取失败: {e}")

    # 经验检测 open 含义
    if not a50_daily.empty and "a50_open" in a50_daily.columns:
        convention = _detect_open_convention(a50_daily, "a50_open", "a50_close")
        print(f"  [检测] A50 open 价格含义: {convention}")
        if convention == "night_session_open":
            print("  [提示] open 为夜盘开盘价，night_ret 将接近0，尝试获取日盘开盘价 ...")

    # 策略2: 用 yfinance 小时线提取真正的日盘开盘价(09:00)
    try:
        import yfinance as yf
        print("  [尝试] 通过 yfinance 获取 A50 小时线数据 ...")
        ticker = yf.Ticker("CN=F")  # FTSE China A50 Futures
        # yfinance 小时线最多 ~730 天历史
        hourly = ticker.history(period="max", interval="1h")
        if not hourly.empty:
            hourly = hourly.reset_index()
            dt_col = "Datetime" if "Datetime" in hourly.columns else "Date"
            hourly["date"] = pd.to_datetime(hourly[dt_col]).dt.tz_localize(None)
            # 提取日盘开盘价: 09:00-09:59 的第一根 bar
            day_open = hourly[hourly["date"].dt.hour == 9].copy()
            day_open["trade_date"] = day_open["date"].dt.date
            day_open = day_open.groupby("trade_date").first().reset_index()
            day_open["date"] = pd.to_datetime(day_open["trade_date"])
            day_open = day_open[["date", "Open"]].rename(columns={"Open": "a50_day_open"})

            # 合并到日线数据，用 a50_day_open 替换 a50_open
            if not a50_daily.empty:
                merged = pd.merge(a50_daily, day_open, on="date", how="left")
                has_day_open = merged["a50_day_open"].notna().sum()
                print(f"  [yfinance] 获取到 {has_day_open} 天的日盘真实开盘价")
                # 有日盘开盘价的行用真实值替换
                mask = merged["a50_day_open"].notna()
                merged.loc[mask, "a50_open"] = merged.loc[mask, "a50_day_open"]
                merged = merged.drop(columns=["a50_day_open"])
                return merged
            else:
                # 没有日线数据，仅用 yfinance
                daily_from_hourly = hourly.groupby(
                    hourly["date"].dt.date
                ).agg({"Open": "first", "Close": "last"}).reset_index()
                daily_from_hourly.columns = ["date", "a50_open", "a50_close"]
                daily_from_hourly["date"] = pd.to_datetime(daily_from_hourly["date"])
                return daily_from_hourly.sort_values("date").reset_index(drop=True)
    except ImportError:
        print("  [提示] yfinance 未安装，跳过小时线数据获取")
    except Exception as e:
        print(f"  [警告] yfinance A50数据获取失败: {e}")

    if not a50_daily.empty:
        return a50_daily

    print(f"  [警告] 获取A50期货数据失败")
    return pd.DataFrame()


def fetch_commodity_futures(symbol: str, name: str) -> pd.DataFrame:
    """获取境外大宗商品期货数据。"""
    print(f"  获取 {name} ({symbol}) 期货数据 ...")
    try:
        df = ak.futures_foreign_hist(symbol=symbol)
        df = df.rename(columns={
            "date": "date", "日期": "date",
            "收盘价": f"cmd_{symbol}_close", "close": f"cmd_{symbol}_close",
            "开盘价": f"cmd_{symbol}_open", "open": f"cmd_{symbol}_open",
        })
        df["date"] = pd.to_datetime(df["date"])
        keep_cols = ["date", f"cmd_{symbol}_close", f"cmd_{symbol}_open"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取 {name} 数据失败: {e}")
        return pd.DataFrame()


def fetch_vix() -> pd.DataFrame:
    """获取VIX恐慌指数数据。"""
    print("  获取VIX恐慌指数数据 ...")
    try:
        df = ak.index_us_stock_sina(symbol=".VIX")
        df = df.rename(columns={"date": "date", "close": "vix_close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "vix_close"]].sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取VIX数据失败: {e}")
        return pd.DataFrame()


def fetch_us_treasury_yield() -> pd.DataFrame:
    """获取美国国债收益率数据。"""
    print("  获取美国10年期国债收益率数据 ...")
    try:
        df = ak.bond_zh_us_rate(start_date="20180101")
        df = df.rename(columns={
            "日期": "date",
            "美国国债收益率10年": "us10y_yield",
            "美国国债收益率2年": "us2y_yield",
        })
        df["date"] = pd.to_datetime(df["date"])
        keep_cols = ["date", "us10y_yield", "us2y_yield"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df["us10y_yield"] = pd.to_numeric(df["us10y_yield"], errors="coerce")
        if "us2y_yield" in df.columns:
            df["us2y_yield"] = pd.to_numeric(df["us2y_yield"], errors="coerce")
            df["us_term_spread"] = df["us10y_yield"] - df["us2y_yield"]
        df = df.dropna(subset=["us10y_yield"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取美债收益率数据失败: {e}")
        return pd.DataFrame()


def fetch_offshore_cnh() -> pd.DataFrame:
    """获取离岸人民币 (USD/CNH) 汇率数据。

    离岸人民币24小时交易，能反映A股休市期间的市场情绪变化。
    与在岸 USD/CNY 不同，CNH 不受央行直接管控，更市场化。
    夜间的 CNH 走势是A股次日开盘的重要参考。
    """
    print("  获取离岸人民币 (USD/CNH) 数据 ...")
    try:
        df = ak.currency_boc_safe(
            symbol="美元",
            start_date="20180101",
            end_date=pd.Timestamp.now().strftime("%Y%m%d"),
        )
        # 中行卖出价更接近离岸市场报价
        df = df.rename(columns={"日期": "date", "中行钞卖价": "usd_cnh"})
        df["date"] = pd.to_datetime(df["date"])
        df["usd_cnh"] = pd.to_numeric(df["usd_cnh"], errors="coerce")
        df = df[["date", "usd_cnh"]].dropna().sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取离岸人民币数据失败: {e}")
        return pd.DataFrame()


def fetch_euro_index(symbol: str, name: str) -> pd.DataFrame:
    """获取欧洲主要指数日线数据。

    欧洲市场交易时间：北京时间 15:00/16:00 - 23:30/00:30+1
    收盘时间早于美股，但晚于A股，可作为A股次日的隔夜参考。
    """
    print(f"  获取 {name} ({symbol}) 数据 ...")
    try:
        # akshare 部分欧洲指数可通过外盘期货接口获取
        df = ak.futures_foreign_hist(symbol=symbol)
        df = df.rename(columns={
            "date": "date", "日期": "date",
            "收盘价": f"eu_{symbol}_close", "close": f"eu_{symbol}_close",
            "开盘价": f"eu_{symbol}_open", "open": f"eu_{symbol}_open",
        })
        df["date"] = pd.to_datetime(df["date"])
        keep_cols = ["date", f"eu_{symbol}_close", f"eu_{symbol}_open"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取 {name} 数据失败: {e}")
        return pd.DataFrame()


def fetch_shibor() -> pd.DataFrame:
    """获取 SHIBOR 利率数据。"""
    print("  获取 SHIBOR 利率数据 ...")
    try:
        df = ak.rate_interbank(market="上海银行间同业拆放利率(Shibor)", symbol="隔夜", indicator="利率")
        df = df.rename(columns={"报告日": "date", "利率": "shibor_on"})
        df["date"] = pd.to_datetime(df["date"])
        df["shibor_on"] = pd.to_numeric(df["shibor_on"], errors="coerce")
        df = df[["date", "shibor_on"]].dropna().sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取 SHIBOR 数据失败: {e}")
        return pd.DataFrame()


def fetch_all_data(start_date: str = "20180101") -> dict:
    """获取所有数据并返回字典。"""
    data = {}

    # 1. A股指数数据
    print("[1/10] 获取A股指数数据 ...")
    for name, code in INDEX_CODES.items():
        data[name] = fetch_index_daily(code, name, start_date)
        time.sleep(0.5)

    # 2. 美股指数
    print("[2/10] 获取美股指数数据 ...")
    overseas_dfs = []
    for name, symbol in OVERSEAS_INDICES.items():
        df = fetch_us_index_daily(symbol, name)
        if not df.empty:
            overseas_dfs.append(df)
        time.sleep(0.5)
    data["overseas"] = overseas_dfs

    # 3. 港股现货指数（非期货，open = 早盘开盘 09:15，无歧义）
    print("[3/10] 获取港股现货指数数据 ...")
    hk_dfs = []
    for name, symbol in HK_INDICES.items():
        df = fetch_hk_index_daily(symbol, name)
        if not df.empty:
            hk_dfs.append(df)
        time.sleep(0.5)
    data["hk"] = hk_dfs

    # 4. 富时A50期货
    print("[4/10] 获取富时A50期货数据 ...")
    data["a50"] = fetch_a50_futures()
    time.sleep(0.5)

    # 5. 美国上市中国ETF
    print("[5/10] 获取美股中国相关ETF数据 ...")
    china_etf_dfs = []
    for name, symbol in US_CHINA_ETFS.items():
        df = fetch_us_china_etf(symbol, name)
        if not df.empty:
            china_etf_dfs.append(df)
        time.sleep(0.5)
    data["china_etfs"] = china_etf_dfs

    # 6. 欧洲指数（收盘于北京时间次日 ~00:30-01:00）
    print("[6/10] 获取欧洲指数数据 ...")
    euro_dfs = []
    for name, symbol in EURO_INDICES.items():
        df = fetch_euro_index(symbol, name)
        if not df.empty:
            euro_dfs.append(df)
        time.sleep(0.5)
    data["euro"] = euro_dfs

    # 7. 大宗商品、汇率、VIX
    print("[7/10] 获取大宗商品、汇率、VIX数据 ...")
    data["gold"] = fetch_gold_price()
    time.sleep(0.5)
    data["usd_cny"] = fetch_usd_cny()
    time.sleep(0.5)
    data["usd_cnh"] = fetch_offshore_cnh()
    time.sleep(0.5)

    commodity_dfs = []
    for name, symbol in COMMODITY_FUTURES.items():
        df = fetch_commodity_futures(symbol, name)
        if not df.empty:
            commodity_dfs.append(df)
        time.sleep(0.5)
    data["commodities"] = commodity_dfs

    data["vix"] = fetch_vix()
    time.sleep(0.5)

    # 8. 美债收益率
    print("[8/10] 获取美债收益率数据 ...")
    data["us_treasury"] = fetch_us_treasury_yield()
    time.sleep(0.5)

    # 9. 利率数据
    print("[9/10] 获取利率数据 ...")
    data["shibor"] = fetch_shibor()

    return data
