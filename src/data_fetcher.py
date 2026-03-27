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

# 港股期货（夜盘交易至次日01:00北京时间，比现货更好的隔夜参考）
HK_FUTURES = {
    "恒指期货": "HSI",      # 恒生指数期货，夜盘至01:00
    "恒生科技期货": "HSTECH",  # 恒生科技指数期货
    "国企指数期货": "HSCEI",   # H股指数期货
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


def fetch_hk_futures_daily(symbol: str, name: str) -> pd.DataFrame:
    """获取港股期货日线数据。

    港股期货夜盘交易至次日01:00（北京时间），比港股现货更好的隔夜参考。
    夜盘收盘价反映了美股开盘后对港股/中概股的最新定价。
    """
    print(f"  获取 {name} 期货 ({symbol}) 数据 ...")
    try:
        # 尝试通过港股指数接口获取
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


def fetch_a50_futures() -> pd.DataFrame:
    """获取富时中国A50期货数据（新加坡交易所）。"""
    print("  获取富时A50期货数据 ...")
    try:
        df = ak.futures_foreign_hist(symbol="A50")
        df = df.rename(columns={
            "date": "date", "日期": "date",
            "收盘价": "a50_close", "close": "a50_close",
            "开盘价": "a50_open", "open": "a50_open",
        })
        df["date"] = pd.to_datetime(df["date"])
        keep_cols = ["date", "a50_close", "a50_open"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [警告] 获取A50期货数据失败: {e}")
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
    print("[1/7] 获取A股指数数据 ...")
    for name, code in INDEX_CODES.items():
        data[name] = fetch_index_daily(code, name, start_date)
        time.sleep(0.5)

    # 2. 美股指数
    print("[2/7] 获取美股指数数据 ...")
    overseas_dfs = []
    for name, symbol in OVERSEAS_INDICES.items():
        df = fetch_us_index_daily(symbol, name)
        if not df.empty:
            overseas_dfs.append(df)
        time.sleep(0.5)
    data["overseas"] = overseas_dfs

    # 3. 港股期货（夜盘交易至01:00，比现货更好的隔夜参考）
    print("[3/8] 获取港股期货数据 ...")
    hk_dfs = []
    for name, symbol in HK_FUTURES.items():
        df = fetch_hk_futures_daily(symbol, name)
        if not df.empty:
            hk_dfs.append(df)
        time.sleep(0.5)
    data["hk"] = hk_dfs

    # 4. 富时A50期货（夜盘至04:45，A股最直接的隔夜参考）
    print("[4/8] 获取富时A50期货数据 ...")
    data["a50"] = fetch_a50_futures()
    time.sleep(0.5)

    # 5. 美国上市中国ETF（美股时段交易，反映海外资金对A股的定价）
    print("[5/8] 获取美股中国相关ETF数据 ...")
    china_etf_dfs = []
    for name, symbol in US_CHINA_ETFS.items():
        df = fetch_us_china_etf(symbol, name)
        if not df.empty:
            china_etf_dfs.append(df)
        time.sleep(0.5)
    data["china_etfs"] = china_etf_dfs

    # 6. 大宗商品、汇率、VIX
    print("[6/8] 获取大宗商品、汇率、VIX数据 ...")
    data["gold"] = fetch_gold_price()
    time.sleep(0.5)
    data["usd_cny"] = fetch_usd_cny()
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

    # 7. 美债收益率
    print("[7/8] 获取美债收益率数据 ...")
    data["us_treasury"] = fetch_us_treasury_yield()
    time.sleep(0.5)

    # 8. 利率数据
    print("[8/8] 获取利率数据 ...")
    data["shibor"] = fetch_shibor()

    return data
