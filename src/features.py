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


def _compute_night_session_return(odf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """从日线 open/close 反推夜盘涨跌幅。

    原理：
    - 期货日线 close[d] = 日盘 T session 收盘价（如A50 16:30、恒指 16:00）
    - 期货日线 open[d+1] = 次日 T session 开盘价（如A50 09:00、恒指 09:30）
    - 由于 T session 开盘价 ≈ 前夜 T+1 session 收盘价（间隔极短），
      因此 open[d+1] / close[d] - 1 ≈ 夜盘涨跌幅

    时间线示例（A50期货，北京时间）：
    d日 16:30  T session 收盘 → close[d]
    d日 17:00  T+1 session 开盘（夜盘开始）
    d+1日 04:45  T+1 session 收盘（夜盘结束）
    d+1日 09:00  新的 T session 开盘 → open[d+1] ≈ 夜盘收盘价
    d+1日 09:30  ★ A股开盘 ← 此时 open[d+1] 已知

    所以对于 A股 d+1 日的预测：
    - 日盘涨跌: close[d] / close[d-1] - 1  （已有）
    - 夜盘涨跌: open[d+1] / close[d] - 1   （新增，最有价值的信号！）
    """
    odf = odf.copy()
    close_col = f"{prefix}_close"
    open_col = f"{prefix}_open"

    if close_col not in odf.columns or open_col not in odf.columns:
        return odf

    # 夜盘涨跌幅 = 次日开盘 / 当日收盘 - 1
    # open[d+1] / close[d] → 赋值给 d+1 行（因为 d+1 开盘时已知）
    odf[f"{prefix}_night_ret"] = (odf[open_col] / odf[close_col].shift(1) - 1) * 100

    # 也计算日盘涨跌幅作为参考
    odf[f"{prefix}_day_ret"] = (odf[close_col] / odf[open_col] - 1) * 100

    return odf


def _merge_overnight_df(df: pd.DataFrame, odf: pd.DataFrame, shift_days: int = 1) -> pd.DataFrame:
    """合并隔夜数据，计算涨跌幅并对齐日期。"""
    if odf is None or odf.empty:
        return df
    odf = odf.copy()
    cols = [c for c in odf.columns if c != "date"]
    for col in cols:
        if "close" in col:
            odf[col.replace("close", "ret")] = odf[col].pct_change() * 100
    if shift_days > 0:
        odf["date"] = odf["date"] + pd.Timedelta(days=shift_days)
    return pd.merge_asof(
        df.sort_values("date"),
        odf.sort_values("date"),
        on="date",
        direction="backward",
    )


def add_overnight_features(
    df: pd.DataFrame,
    overseas_dfs: list[pd.DataFrame],
    gold_df: pd.DataFrame,
    usd_cny_df: pd.DataFrame,
    shibor_df: pd.DataFrame,
    hk_dfs: list[pd.DataFrame] | None = None,
    a50_df: pd.DataFrame | None = None,
    commodity_dfs: list[pd.DataFrame] | None = None,
    vix_df: pd.DataFrame | None = None,
    us_treasury_df: pd.DataFrame | None = None,
    china_etf_dfs: list[pd.DataFrame] | None = None,
    euro_dfs: list[pd.DataFrame] | None = None,
    usd_cnh_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """添加隔夜因子特征（海外市场、商品、汇率等）。

    ===== 时间对齐原则 =====
    预测目标: A股 T日 9:30 的开盘跳空
    可用信息: 严格限于 T日 9:30 之前已确定的数据

    各数据源时间线（北京时间）:

    【境内数据源（T日数据在A股开盘后才产生 → 必须 shift+1）】
    - 上海金基准价:    T日白天交易时段产生 → 只能用 T-1日数据
    - SHIBOR隔夜利率:  T日 11:30 发布     → 只能用 T-1日数据
    - USD/CNY汇率:     中行汇买价更新时间不确定 → 保守处理，只用 T-1日数据

    【境外数据源（日历日 d 的数据 → shift+1 对齐到 A股 d+1 日）】
    - 美股三大指数:     收盘于北京时间 d+1 凌晨 ~04:00 ✓
    - 美股中国ETF:      同美股时段 ✓
    - VIX:             同美股时段 ✓
    - 美债收益率:       美国时间日终 ✓
    - 境外商品期货:     以美国时间结算 ✓
    - 欧洲指数:        收盘于北京时间 d+1 凌晨 ~00:30-01:00 ✓
    - 离岸人民币CNH:   24小时交易，日结算以美国时间为准 ✓

    【港股/A50 夜盘数据恢复】
    虽然 akshare 日线只包含日盘收盘价，但可以从 open/close 反推夜盘涨跌：
    - 夜盘涨跌幅 ≈ open[d+1] / close[d] - 1
    - 因为次日 T session 开盘价 ≈ 前夜 T+1 session 收盘价（两个时段间隔极短）
    - 这对 A50期货尤其重要：夜盘(17:00-04:45)跨越了整个美股交易时段

    时间线：
    d日 16:30  A50 日盘收盘 → close[d]
    d日 17:00  A50 夜盘开盘
    d+1日 04:45  A50 夜盘收盘 ← 我们要的信息
    d+1日 09:00  A50 日盘开盘 → open[d+1] ≈ 夜盘收盘价
    d+1日 09:30  ★ A股开盘 ← open[d+1] 此时已知

    对于 A股 d+1 的预测：
    - open[d+1] / close[d] - 1 = A50 夜盘涨跌幅 → 这行数据本身就在 d+1 行
    - 所以 night_ret 不需要额外 shift，它天然对齐到了 d+1 日
    - 但 close/day_ret 还是需要 shift+1（日盘 d 的信息 → 用于 A股 d+1）
    """
    df = df.copy()

    # ============================================================
    # 境外数据源：日历日 d 的数据 shift+1 对齐到 A股 d+1 日
    # ============================================================

    # === 美股指数 ===
    for odf in overseas_dfs:
        df = _merge_overnight_df(df, odf, shift_days=1)

    # === 美股中国相关ETF ===
    if china_etf_dfs:
        for odf in china_etf_dfs:
            df = _merge_overnight_df(df, odf, shift_days=1)

    # === 港股期货（含夜盘涨跌幅反推） ===
    if hk_dfs:
        for odf in hk_dfs:
            # 识别 prefix
            close_cols = [c for c in odf.columns if c.endswith("_close")]
            if close_cols:
                prefix = close_cols[0].replace("_close", "")
                odf = _compute_night_session_return(odf, prefix)
            df = _merge_overnight_df(df, odf, shift_days=1)

    # === 富时A50期货（含夜盘涨跌幅反推 — 最有价值的隔夜信号） ===
    if a50_df is not None and not a50_df.empty:
        a50 = a50_df.copy()
        a50 = _compute_night_session_return(a50, "a50")
        a50["a50_ret"] = a50["a50_close"].pct_change() * 100

        # 分离夜盘特征（不需要额外shift）和日盘特征（需要shift+1）
        # night_ret[d+1] = open[d+1]/close[d] - 1，对于A股 d+1 日预测已对齐
        # 但经过 shift+1 后，A股 d+1 会匹配到 d+1 的数据 → night_ret 正确
        a50["date"] = a50["date"] + pd.Timedelta(days=1)
        df = pd.merge_asof(
            df.sort_values("date"),
            a50.sort_values("date"),
            on="date",
            direction="backward",
        )

    # === 境外商品期货（含夜盘涨跌幅反推） ===
    if commodity_dfs:
        for odf in commodity_dfs:
            close_cols = [c for c in odf.columns if c.endswith("_close")]
            if close_cols:
                prefix = close_cols[0].replace("_close", "")
                odf = _compute_night_session_return(odf, prefix)
            df = _merge_overnight_df(df, odf, shift_days=1)

    # === 欧洲指数（收盘于北京时间 d+1 ~00:30-01:00） ===
    if euro_dfs:
        for odf in euro_dfs:
            df = _merge_overnight_df(df, odf, shift_days=1)

    # === VIX 恐慌指数 ===
    if vix_df is not None and not vix_df.empty:
        vix = vix_df.copy()
        vix["vix_ret"] = vix["vix_close"].pct_change() * 100
        vix["vix_level"] = vix["vix_close"]
        vix["date"] = vix["date"] + pd.Timedelta(days=1)
        df = pd.merge_asof(
            df.sort_values("date"),
            vix[["date", "vix_close", "vix_ret", "vix_level"]].sort_values("date"),
            on="date",
            direction="backward",
        )

    # === 美债收益率 ===
    if us_treasury_df is not None and not us_treasury_df.empty:
        tsy = us_treasury_df.copy()
        if "us10y_yield" in tsy.columns:
            tsy["us10y_yield_chg"] = tsy["us10y_yield"].diff()
        if "us_term_spread" in tsy.columns:
            tsy["term_spread_chg"] = tsy["us_term_spread"].diff()
        tsy["date"] = tsy["date"] + pd.Timedelta(days=1)
        df = pd.merge_asof(
            df.sort_values("date"),
            tsy.sort_values("date"),
            on="date",
            direction="backward",
        )

    # === 离岸人民币 CNH（24小时交易，shift+1 用日结算数据） ===
    if usd_cnh_df is not None and not usd_cnh_df.empty:
        cnh = usd_cnh_df.copy()
        cnh["usd_cnh_ret"] = cnh["usd_cnh"].pct_change() * 100
        # CNH 与在岸 CNY 的价差也有信息量
        cnh["date"] = cnh["date"] + pd.Timedelta(days=1)
        df = pd.merge_asof(
            df.sort_values("date"),
            cnh.sort_values("date"),
            on="date",
            direction="backward",
        )

    # ============================================================
    # 境内数据源：T日数据在 A股 T日 9:30 开盘后才产生 → shift+1
    # ============================================================

    # === 黄金（上海金基准价） ===
    if gold_df is not None and not gold_df.empty:
        gold = gold_df.copy()
        gold["gold_ret"] = gold["gold_price"].pct_change() * 100
        gold["date"] = gold["date"] + pd.Timedelta(days=1)
        df = pd.merge_asof(
            df.sort_values("date"),
            gold.sort_values("date"),
            on="date",
            direction="backward",
        )

    # === 在岸汇率（中行汇买价） ===
    if usd_cny_df is not None and not usd_cny_df.empty:
        fx = usd_cny_df.copy()
        fx["usd_cny_ret"] = fx["usd_cny"].pct_change() * 100
        fx["date"] = fx["date"] + pd.Timedelta(days=1)
        df = pd.merge_asof(
            df.sort_values("date"),
            fx.sort_values("date"),
            on="date",
            direction="backward",
        )

    # 在岸/离岸价差 (CNH - CNY spread)
    if "usd_cnh" in df.columns and "usd_cny" in df.columns:
        df["cnh_cny_spread"] = df["usd_cnh"] - df["usd_cny"]

    # === SHIBOR（T日 11:30 发布） ===
    if shibor_df is not None and not shibor_df.empty:
        shibor = shibor_df.copy()
        shibor["date"] = shibor["date"] + pd.Timedelta(days=1)
        df = pd.merge_asof(
            df.sort_values("date"),
            shibor.sort_values("date"),
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
    hk_dfs: list[pd.DataFrame] | None = None,
    a50_df: pd.DataFrame | None = None,
    commodity_dfs: list[pd.DataFrame] | None = None,
    vix_df: pd.DataFrame | None = None,
    us_treasury_df: pd.DataFrame | None = None,
    china_etf_dfs: list[pd.DataFrame] | None = None,
    euro_dfs: list[pd.DataFrame] | None = None,
    usd_cnh_df: pd.DataFrame | None = None,
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
    df = add_overnight_features(
        df, overseas_dfs, gold_df, usd_cny_df, shibor_df,
        hk_dfs=hk_dfs, a50_df=a50_df, commodity_dfs=commodity_dfs,
        vix_df=vix_df, us_treasury_df=us_treasury_df,
        china_etf_dfs=china_etf_dfs,
        euro_dfs=euro_dfs, usd_cnh_df=usd_cnh_df,
    )

    # ============================================================
    # 5. 时间对齐 shift 处理
    # ============================================================
    #
    # 预测目标: A股 T日 9:30 开盘跳空
    # 可用信息: T日 9:30 之前已确定的数据
    #
    # 【A股自身技术指标】
    # 这些指标使用了 T日 的 OHLCV 计算（如 close, volume, turnover），
    # 而 T日 的 OHLCV 在 T日 15:00 收盘后才确定。
    # 因此必须 shift(1)，将 T-1 日的指标值用于 T日 的预测。
    #
    # 【隔夜因子】
    # 已在 add_overnight_features() 中通过日期 +1 天处理，无需额外 shift。
    # - 境外数据(美股/港股/A50/商品/VIX/美债): date += 1 day
    # - 境内数据(黄金/汇率/SHIBOR): date += 1 day
    #
    # 【日历特征】
    # weekday/month/quarter/is_month_start/is_month_end: T日的日历属性
    # 在 T日 开盘前就已知，无需 shift。
    # days_since_prev/is_after_holiday: 基于交易日间隔，T日开盘前已知，无需 shift。

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
