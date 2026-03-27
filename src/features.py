"""
特征工程模块：构建用于预测开盘跳空幅度的特征。
"""
import numpy as np
import pandas as pd


def compute_gap(df: pd.DataFrame) -> pd.DataFrame:
    """计算开盘跳空幅度（目标变量）: gap = (open_t - close_{t-1}) / close_{t-1} * 100。"""
    df = df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["gap_pct"] = (df["open"] - df["prev_close"]) / df["prev_close"] * 100
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加技术指标特征（均使用前一日数据，避免未来信息泄露）。"""
    df = df.copy()

    # 收益率
    df["ret_1d"] = df["close"].pct_change(1) * 100
    df["ret_3d"] = df["close"].pct_change(3) * 100
    df["ret_5d"] = df["close"].pct_change(5) * 100
    df["ret_10d"] = df["close"].pct_change(10) * 100
    df["ret_20d"] = df["close"].pct_change(20) * 100

    # 波动率
    df["volatility_5d"] = df["ret_1d"].rolling(5).std()
    df["volatility_10d"] = df["ret_1d"].rolling(10).std()
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()

    # 均线偏离度
    for w in [5, 10, 20, 60]:
        ma = df["close"].rolling(w).mean()
        df[f"ma_bias_{w}d"] = (df["close"] - ma) / ma * 100

    # 成交量变化率
    df["vol_ratio_5d"] = df["volume"] / df["volume"].rolling(5).mean()
    df["vol_ratio_10d"] = df["volume"] / df["volume"].rolling(10).mean()

    # 成交额变化率
    if "amount" in df.columns:
        df["amt_ratio_5d"] = df["amount"] / df["amount"].rolling(5).mean()

    # 振幅
    df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100

    # 上下影线
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"] * 100
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"] * 100

    # 日内涨跌（开盘到收盘）
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"] * 100

    # 前一日跳空
    df["prev_gap"] = df["gap_pct"].shift(1) if "gap_pct" in df.columns else np.nan

    # 换手率相关
    if "turnover" in df.columns:
        df["turnover_ma5"] = df["turnover"].rolling(5).mean()
        df["turnover_ratio"] = df["turnover"] / df["turnover"].rolling(10).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 布林带位置
    bb_ma = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - bb_ma) / (2 * bb_std)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加日历效应特征。"""
    df = df.copy()
    df["weekday"] = df["date"].dt.weekday  # 0=Monday
    df["month"] = df["date"].dt.month
    df["is_month_start"] = (df["date"].dt.day <= 3).astype(int)
    df["is_month_end"] = (df["date"].dt.day >= 25).astype(int)

    # 距离上一个交易日的天数（检测长假效应）
    df["days_since_prev"] = df["date"].diff().dt.days
    df["is_after_holiday"] = (df["days_since_prev"] > 3).astype(int)

    # 季度
    df["quarter"] = df["date"].dt.quarter

    return df


def add_overnight_features(
    df: pd.DataFrame,
    overseas_dfs: list[pd.DataFrame],
    gold_df: pd.DataFrame,
    usd_cny_df: pd.DataFrame,
    shibor_df: pd.DataFrame,
) -> pd.DataFrame:
    """添加隔夜因子特征（海外市场、商品、汇率等）。

    关键：A股隔夜期间，美股是正常交易时段。
    用美股当日收盘数据对齐到A股次日（即A股开盘前已知的信息）。
    """
    df = df.copy()

    # 合并海外市场数据
    for odf in overseas_dfs:
        if odf.empty:
            continue
        # 美股 T 日收盘 → 对应 A股 T+1 日开盘前信息
        odf = odf.copy()
        cols = [c for c in odf.columns if c != "date"]
        # 计算美股涨跌幅
        for col in cols:
            if "close" in col:
                odf[col.replace("close", "ret")] = odf[col].pct_change() * 100
        odf["date"] = odf["date"] + pd.Timedelta(days=1)  # 移到下一日对齐
        df = pd.merge_asof(
            df.sort_values("date"),
            odf.sort_values("date"),
            on="date",
            direction="backward",
        )

    # 黄金
    if gold_df is not None and not gold_df.empty:
        gold = gold_df.copy()
        gold["gold_ret"] = gold["gold_price"].pct_change() * 100
        df = pd.merge_asof(
            df.sort_values("date"),
            gold.sort_values("date"),
            on="date",
            direction="backward",
        )

    # 汇率
    if usd_cny_df is not None and not usd_cny_df.empty:
        fx = usd_cny_df.copy()
        fx["usd_cny_ret"] = fx["usd_cny"].pct_change() * 100
        df = pd.merge_asof(
            df.sort_values("date"),
            fx.sort_values("date"),
            on="date",
            direction="backward",
        )

    # SHIBOR
    if shibor_df is not None and not shibor_df.empty:
        df = pd.merge_asof(
            df.sort_values("date"),
            shibor_df.sort_values("date"),
            on="date",
            direction="backward",
        )

    return df


def build_features(
    index_df: pd.DataFrame,
    overseas_dfs: list[pd.DataFrame],
    gold_df: pd.DataFrame,
    usd_cny_df: pd.DataFrame,
    shibor_df: pd.DataFrame,
) -> pd.DataFrame:
    """构建完整特征集。"""
    df = index_df.copy()

    # 1. 计算目标变量
    df = compute_gap(df)

    # 2. 技术指标
    df = add_technical_features(df)

    # 3. 日历效应
    df = add_calendar_features(df)

    # 4. 隔夜因子
    df = add_overnight_features(df, overseas_dfs, gold_df, usd_cny_df, shibor_df)

    # 5. 所有特征向后移一期（确保使用前一日信息预测当日开盘跳空）
    target_col = "gap_pct"
    date_col = "date"
    non_feature_cols = [date_col, target_col, "open", "close", "high", "low",
                        "volume", "amount", "prev_close", "turnover",
                        "pct_change", "change"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # 对于技术指标，它们已经基于当日收盘价计算，需要shift(1)确保不泄露
    # 但隔夜因子（overseas已经+1天对齐）不需要额外shift
    # 统一处理：所有自行计算的技术指标和日历特征用当日值（它们基于历史数据计算）
    # 隔夜因子已通过日期偏移处理

    # shift 技术指标和日历特征（这些使用了当日收盘价信息）
    tech_cols = [c for c in df.columns if any(c.startswith(p) for p in [
        "ret_", "volatility_", "ma_bias_", "vol_ratio_", "amt_ratio_",
        "amplitude", "upper_shadow", "lower_shadow", "intraday_ret",
        "prev_gap", "turnover_ma5", "turnover_ratio", "rsi_", "macd",
        "bb_position",
    ])]
    for col in tech_cols:
        df[col] = df[col].shift(1)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """获取特征列名列表。"""
    exclude_cols = {"date", "gap_pct", "open", "close", "high", "low",
                    "volume", "amount", "prev_close", "turnover",
                    "pct_change", "change"}
    return [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
