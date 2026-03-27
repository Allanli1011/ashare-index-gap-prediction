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
    print("[1/4] 获取A股指数数据 ...")
    for name, code in INDEX_CODES.items():
        data[name] = fetch_index_daily(code, name, start_date)
        time.sleep(0.5)

    # 2. 海外市场数据
    print("[2/4] 获取海外市场数据 ...")
    overseas_dfs = []
    for name, symbol in OVERSEAS_INDICES.items():
        df = fetch_us_index_daily(symbol, name)
        if not df.empty:
            overseas_dfs.append(df)
        time.sleep(0.5)
    data["overseas"] = overseas_dfs

    # 3. 商品和汇率
    print("[3/4] 获取商品和汇率数据 ...")
    data["gold"] = fetch_gold_price()
    time.sleep(0.5)
    data["usd_cny"] = fetch_usd_cny()
    time.sleep(0.5)

    # 4. 利率数据
    print("[4/4] 获取利率数据 ...")
    data["shibor"] = fetch_shibor()

    return data
