"""
合成数据生成模块：当无法访问 akshare API 时，生成模拟数据用于演示和测试。
生成的数据模拟A股指数和海外市场的统计特征。
"""
import numpy as np
import pandas as pd


def generate_index_data(
    name: str,
    start_date: str = "2018-01-02",
    end_date: str = "2025-12-31",
    initial_price: float = 3500.0,
    annual_return: float = 0.05,
    annual_vol: float = 0.22,
    seed: int = 42,
) -> pd.DataFrame:
    """生成模拟的A股指数日线数据。"""
    rng = np.random.RandomState(seed)

    # 生成交易日序列（排除周末）
    all_dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    # 模拟中国节假日：随机删除一些日期
    holiday_mask = rng.random(len(all_dates)) > 0.03  # ~3%的天是假期
    dates = all_dates[holiday_mask]
    n = len(dates)

    # 日收益率参数
    daily_mu = annual_return / 252
    daily_sigma = annual_vol / np.sqrt(252)

    # 生成收盘价序列（GBM + 均值回复 + 跳跃）
    log_returns = rng.normal(daily_mu, daily_sigma, n)

    # 添加跳跃
    jump_prob = 0.02
    jump_mask = rng.random(n) < jump_prob
    jump_sizes = rng.normal(0, daily_sigma * 3, n) * jump_mask
    log_returns += jump_sizes

    # 添加自相关（短期动量 + 长期反转）
    for i in range(1, n):
        log_returns[i] += 0.05 * log_returns[i - 1]  # 短期动量

    close_prices = initial_price * np.exp(np.cumsum(log_returns))

    # 生成开盘价（包含跳空）
    gap_returns = rng.normal(0, daily_sigma * 0.3, n)
    # 跳空与前日涨跌有弱相关
    for i in range(1, n):
        gap_returns[i] += -0.1 * log_returns[i - 1]  # 弱反转效应
    open_prices = np.zeros(n)
    open_prices[0] = close_prices[0] * (1 + gap_returns[0])
    for i in range(1, n):
        open_prices[i] = close_prices[i - 1] * (1 + gap_returns[i])

    # 生成最高价和最低价
    intraday_range = np.abs(rng.normal(0, daily_sigma, n)) + 0.002
    high_prices = np.maximum(open_prices, close_prices) * (1 + intraday_range * 0.5)
    low_prices = np.minimum(open_prices, close_prices) * (1 - intraday_range * 0.5)

    # 成交量（与波动率正相关）
    base_volume = 1e9
    vol_factor = 1 + 2 * np.abs(log_returns) / daily_sigma
    volume = base_volume * vol_factor * (1 + rng.normal(0, 0.3, n))
    volume = np.maximum(volume, base_volume * 0.3)

    # 成交额
    amount = volume * (close_prices + open_prices) / 2

    # 换手率
    turnover = 0.5 + rng.exponential(0.3, n)

    df = pd.DataFrame({
        "date": dates[:n],
        "open": np.round(open_prices, 2),
        "close": np.round(close_prices, 2),
        "high": np.round(high_prices, 2),
        "low": np.round(low_prices, 2),
        "volume": volume.astype(int),
        "amount": np.round(amount, 2),
        "turnover": np.round(turnover, 2),
    })

    # 振幅和涨跌
    df["amplitude"] = np.round((df["high"] - df["low"]) / df["close"].shift(1) * 100, 2)
    df["pct_change"] = np.round(df["close"].pct_change() * 100, 2)
    df["change"] = np.round(df["close"].diff(), 2)

    print(f"  [模拟数据] {name}: {len(df)} 条记录, "
          f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    return df


def generate_us_index_data(
    symbol: str,
    start_date: str = "2018-01-02",
    end_date: str = "2025-12-31",
    initial_price: float = 2700.0,
    seed: int = 100,
) -> pd.DataFrame:
    """生成模拟的美股指数数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    daily_sigma = 0.20 / np.sqrt(252)
    daily_mu = 0.08 / 252
    log_returns = rng.normal(daily_mu, daily_sigma, n)
    close_prices = initial_price * np.exp(np.cumsum(log_returns))
    open_prices = close_prices * (1 + rng.normal(0, daily_sigma * 0.2, n))

    df = pd.DataFrame({
        "date": dates[:n],
        f"us_{symbol}_close": np.round(close_prices, 2),
        f"us_{symbol}_open": np.round(open_prices, 2),
    })
    return df


def generate_gold_data(
    start_date: str = "2018-01-02",
    end_date: str = "2025-12-31",
    seed: int = 200,
) -> pd.DataFrame:
    """生成模拟黄金价格数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    daily_sigma = 0.15 / np.sqrt(252)
    log_returns = rng.normal(0.03 / 252, daily_sigma, n)
    prices = 280 * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame({
        "date": dates[:n],
        "gold_price": np.round(prices, 2),
    })
    return df


def generate_fx_data(
    start_date: str = "2018-01-02",
    end_date: str = "2025-12-31",
    seed: int = 300,
) -> pd.DataFrame:
    """生成模拟美元兑人民币汇率数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    daily_sigma = 0.05 / np.sqrt(252)
    log_returns = rng.normal(0, daily_sigma, n)
    rates = 6.8 * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame({
        "date": dates[:n],
        "usd_cny": np.round(rates, 4),
    })
    return df


def generate_shibor_data(
    start_date: str = "2018-01-02",
    end_date: str = "2025-12-31",
    seed: int = 400,
) -> pd.DataFrame:
    """生成模拟SHIBOR隔夜利率数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    # 均值回复过程
    rate = 2.0
    rates = []
    for _ in range(n):
        rate += 0.1 * (2.0 - rate) + rng.normal(0, 0.15)
        rate = max(0.5, min(rate, 6.0))
        rates.append(rate)

    df = pd.DataFrame({
        "date": dates[:n],
        "shibor_on": np.round(rates, 4),
    })
    return df


def generate_generic_asset_data(
    prefix: str,
    start_date: str = "2018-01-02",
    end_date: str = "2025-12-31",
    initial_price: float = 1000.0,
    annual_vol: float = 0.20,
    seed: int = 500,
) -> pd.DataFrame:
    """生成通用资产价格数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    daily_sigma = annual_vol / np.sqrt(252)
    log_returns = rng.normal(0.05 / 252, daily_sigma, n)
    close_prices = initial_price * np.exp(np.cumsum(log_returns))
    open_prices = close_prices * (1 + rng.normal(0, daily_sigma * 0.2, n))

    df = pd.DataFrame({
        "date": dates[:n],
        f"{prefix}_close": np.round(close_prices, 2),
        f"{prefix}_open": np.round(open_prices, 2),
    })
    return df


def generate_vix_data(
    start_date: str = "2018-01-02",
    end_date: str = "2025-12-31",
    seed: int = 600,
) -> pd.DataFrame:
    """生成模拟VIX数据（均值回复过程）。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    vix = 18.0
    vix_values = []
    for _ in range(n):
        vix += 0.05 * (18.0 - vix) + rng.normal(0, 1.5)
        vix = max(9.0, min(vix, 80.0))
        vix_values.append(vix)

    return pd.DataFrame({
        "date": dates[:n],
        "vix_close": np.round(vix_values, 2),
    })


def generate_treasury_data(
    start_date: str = "2018-01-02",
    end_date: str = "2025-12-31",
    seed: int = 700,
) -> pd.DataFrame:
    """生成模拟美债收益率数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    y10 = 2.8
    y2 = 2.5
    y10_values, y2_values = [], []
    for _ in range(n):
        y10 += 0.02 * (3.0 - y10) + rng.normal(0, 0.05)
        y2 += 0.02 * (2.5 - y2) + rng.normal(0, 0.04)
        y10 = max(0.5, min(y10, 5.5))
        y2 = max(0.2, min(y2, 5.5))
        y10_values.append(y10)
        y2_values.append(y2)

    df = pd.DataFrame({
        "date": dates[:n],
        "us10y_yield": np.round(y10_values, 4),
        "us2y_yield": np.round(y2_values, 4),
    })
    df["us_term_spread"] = np.round(df["us10y_yield"] - df["us2y_yield"], 4)
    return df


def generate_all_data() -> dict:
    """生成所有模拟数据。"""
    from src.data_fetcher import (
        INDEX_CODES, OVERSEAS_INDICES, HK_FUTURES,
        US_CHINA_ETFS, COMMODITY_FUTURES,
    )

    data = {}

    # A股指数
    print("[模拟模式] 生成A股指数数据 ...")
    index_configs = {
        "沪深300": {"initial_price": 3500, "annual_vol": 0.22, "seed": 42},
        "中证500": {"initial_price": 5500, "annual_vol": 0.28, "seed": 43},
        "中证1000": {"initial_price": 6000, "annual_vol": 0.30, "seed": 44},
    }
    for name in INDEX_CODES:
        cfg = index_configs[name]
        data[name] = generate_index_data(name, **cfg)

    # 美股指数
    print("[模拟模式] 生成美股指数数据 ...")
    overseas_dfs = []
    us_configs = {
        "SPX": {"initial_price": 2700, "seed": 100},
        "NDX": {"initial_price": 7000, "seed": 101},
        "DJI": {"initial_price": 25000, "seed": 102},
    }
    for symbol in OVERSEAS_INDICES.values():
        cfg = us_configs[symbol]
        overseas_dfs.append(generate_us_index_data(symbol, **cfg))
    data["overseas"] = overseas_dfs

    # 港股期货
    print("[模拟模式] 生成港股期货数据 ...")
    hk_dfs = []
    hk_configs = {
        "HSI": {"initial_price": 28000, "annual_vol": 0.22, "seed": 110},
        "HSTECH": {"initial_price": 5000, "annual_vol": 0.35, "seed": 111},
        "HSCEI": {"initial_price": 10000, "annual_vol": 0.25, "seed": 112},
    }
    for symbol in HK_FUTURES.values():
        cfg = hk_configs[symbol]
        hk_dfs.append(generate_generic_asset_data(
            prefix=f"hk_{symbol}",
            initial_price=cfg["initial_price"],
            annual_vol=cfg["annual_vol"],
            seed=cfg["seed"],
        ))
    data["hk"] = hk_dfs

    # 富时A50期货
    print("[模拟模式] 生成富时A50期货数据 ...")
    a50 = generate_generic_asset_data(prefix="a50", initial_price=12000, annual_vol=0.24, seed=120)
    data["a50"] = a50

    # 美股中国ETF
    print("[模拟模式] 生成美股中国ETF数据 ...")
    etf_dfs = []
    etf_configs = {
        "FXI": {"initial_price": 42, "annual_vol": 0.28, "seed": 130},
        "KWEB": {"initial_price": 55, "annual_vol": 0.38, "seed": 131},
        "ASHR": {"initial_price": 28, "annual_vol": 0.25, "seed": 132},
        "MCHI": {"initial_price": 65, "annual_vol": 0.28, "seed": 133},
    }
    for symbol in US_CHINA_ETFS.values():
        cfg = etf_configs[symbol]
        etf_dfs.append(generate_generic_asset_data(
            prefix=f"etf_{symbol}",
            initial_price=cfg["initial_price"],
            annual_vol=cfg["annual_vol"],
            seed=cfg["seed"],
        ))
    data["china_etfs"] = etf_dfs

    # 大宗商品
    print("[模拟模式] 生成大宗商品数据 ...")
    cmd_dfs = []
    cmd_configs = {
        "CL": {"initial_price": 65, "annual_vol": 0.35, "seed": 140},
        "OIL": {"initial_price": 70, "annual_vol": 0.33, "seed": 141},
        "HG": {"initial_price": 3.0, "annual_vol": 0.25, "seed": 142},
    }
    for symbol in COMMODITY_FUTURES.values():
        cfg = cmd_configs[symbol]
        cmd_dfs.append(generate_generic_asset_data(
            prefix=f"cmd_{symbol}",
            initial_price=cfg["initial_price"],
            annual_vol=cfg["annual_vol"],
            seed=cfg["seed"],
        ))
    data["commodities"] = cmd_dfs

    # VIX
    print("[模拟模式] 生成VIX数据 ...")
    data["vix"] = generate_vix_data()

    # 美债收益率
    print("[模拟模式] 生成美债收益率数据 ...")
    data["us_treasury"] = generate_treasury_data()

    # 黄金、汇率、SHIBOR
    print("[模拟模式] 生成商品、汇率、利率数据 ...")
    data["gold"] = generate_gold_data()
    data["usd_cny"] = generate_fx_data()
    data["shibor"] = generate_shibor_data()

    return data
