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


def generate_all_data() -> dict:
    """生成所有模拟数据。"""
    from src.data_fetcher import INDEX_CODES, OVERSEAS_INDICES

    data = {}

    print("[模拟模式] 生成A股指数数据 ...")
    index_configs = {
        "沪深300": {"initial_price": 3500, "annual_vol": 0.22, "seed": 42},
        "中证500": {"initial_price": 5500, "annual_vol": 0.28, "seed": 43},
        "中证1000": {"initial_price": 6000, "annual_vol": 0.30, "seed": 44},
    }
    for name in INDEX_CODES:
        cfg = index_configs[name]
        data[name] = generate_index_data(name, **cfg)

    print("[模拟模式] 生成海外市场数据 ...")
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

    print("[模拟模式] 生成商品和汇率数据 ...")
    data["gold"] = generate_gold_data()
    data["usd_cny"] = generate_fx_data()

    print("[模拟模式] 生成利率数据 ...")
    data["shibor"] = generate_shibor_data()

    return data
