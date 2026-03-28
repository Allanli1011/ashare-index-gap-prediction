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


def _merge_futures_with_night_session(
    df: pd.DataFrame,
    odf: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """合并期货数据，分开处理日盘和夜盘特征以确保正确的时间对齐。

    核心问题：
    期货数据源的日线 open 价格含义不确定：
    - 情况A: open = 日盘开盘价（如 A50 09:00，恒指 09:15）
    - 情况B: open = 夜盘开盘价（前一天傍晚，如 A50 17:00，恒指 17:15）

    如果是情况B，之前的公式 open[d+1]/close[d]-1 只是日盘收盘到夜盘开盘的
    30分钟~1小时价差（≈0%），完全无法衡量隔夜涨跌幅。

    修正方案 — 分离两次 merge：
    1. close-based features（close, ret, day_ret）: shift+1
       日盘收盘(16:00-16:30) 晚于 A股收盘(15:00) → 用于次日预测
    2. open price: 不 shift，直接用当日数据
       - 情况A: 日盘开盘(09:00/09:15) 早于 A股开盘(09:30) → 当天可用 ✓
       - 情况B: 前晚夜盘开盘价 → 更早已知 → 也可用 ✓
    3. night_ret = 当日 open / 前日 close：在 merge 后计算
       - 情况A: 反映了完整隔夜涨跌（最有价值的信号）
       - 情况B: ≈ 0%（不提供信息，但模型会自动忽略）

    时间线（以 A50 为例，情况A）：
    T-1日 16:30  A50 日盘收盘 → close[T-1]  ─┐
    T-1日 17:00  A50 夜盘开盘                   │ 夜盘涨跌幅
    T日   04:45  A50 夜盘收盘                   │ ≈ open[T] / close[T-1] - 1
    T日   09:00  A50 日盘开盘 → open[T]     ─┘
    T日   09:30  ★ A股开盘 ← 此时 open[T] 和 close[T-1] 都已知
    """
    if odf is None or odf.empty:
        return df

    odf = odf.copy()
    close_col = f"{prefix}_close"
    open_col = f"{prefix}_open"

    # --- Part 1: close-based features, shift+1 ---
    if close_col in odf.columns:
        close_feats = odf[["date", close_col]].copy()
        close_feats[f"{prefix}_ret"] = close_feats[close_col].pct_change() * 100
        if open_col in odf.columns:
            close_feats[f"{prefix}_day_ret"] = (
                odf[close_col].values / odf[open_col].values - 1
            ) * 100
        close_feats["date"] = close_feats["date"] + pd.Timedelta(days=1)
        df = pd.merge_asof(
            df.sort_values("date"),
            close_feats.sort_values("date"),
            on="date",
            direction="backward",
        )

    # --- Part 2: open price, NO shift ---
    # 当日的 open 价格（无论是日盘开盘还是前晚夜盘开盘）在 A股 09:30 前已知
    if open_col in odf.columns:
        open_feats = odf[["date", open_col]].copy()
        open_feats = open_feats.rename(columns={open_col: f"{prefix}_open_today"})
        df = pd.merge_asof(
            df.sort_values("date"),
            open_feats.sort_values("date"),
            on="date",
            direction="backward",
        )

        # --- Part 3: night_ret = 当日 open / 前日 close ---
        # close 来自 shift+1 后的数据 = 前一交易日收盘价
        # open_today 来自未 shift 的数据 = 当日开盘价
        # 两者相除即为隔夜涨跌幅
        if close_col in df.columns and f"{prefix}_open_today" in df.columns:
            df[f"{prefix}_night_ret"] = (
                df[f"{prefix}_open_today"] / df[close_col] - 1
            ) * 100
        df = df.drop(columns=[f"{prefix}_open_today"], errors="ignore")

    return df


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

    【港股数据】
    实际使用的是 akshare stock_hk_index_daily_sina → 现货指数（非期货）。
    现货指数无夜盘，open = 早盘开盘(09:15 HKT)，close = 午盘收盘(16:00 HKT)。
    open 在 A股 09:30 前 15 分钟已知 → 可直接用于预测，无歧义。
    night_ret = open[T] / close[T-1] = 精确的隔夜涨跌幅。

    【A50 期货 — open 价格含义不确定】
    SGX 交易日: T+1 session(夜盘 17:00-04:45) + T session(日盘 09:00-16:30)
    akshare 日线 open 大概率 = 夜盘开盘价(前日17:00)，非日盘开盘价(09:00)。

    解决方案：
    1. _merge_futures_with_night_session 分离 open(不shift) 和 close(shift+1)
    2. data_fetcher.py 尝试用 yfinance 小时线获取真实的日盘开盘价
    3. _detect_open_convention 经验检测 open 含义
    4. 无论哪种情况，模型都能正确处理：
       - open = 日盘开盘 → night_ret 反映完整隔夜涨跌 ✓
       - open = 夜盘开盘 → night_ret ≈ 0%，模型自动忽略 ✓
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

    # === 港股现货指数（非期货！open = 早盘开盘 09:15，无歧义） ===
    # 数据来源: akshare stock_hk_index_daily_sina → 现货指数，无夜盘
    # open[T] = 09:15 HKT 开盘价，在 A股 09:30 开盘前已知
    # 使用 _merge_futures_with_night_session 将 open 和 close 分开处理：
    # - close: shift+1（16:00 HKT 收盘 → 用于次日预测）
    # - open: 不shift（09:15 HKT → 同日可用）
    # - night_ret = open[T] / close[T-1]（精确的隔夜涨跌幅）
    if hk_dfs:
        for odf in hk_dfs:
            close_cols = [c for c in odf.columns if c.endswith("_close")]
            if close_cols:
                prefix = close_cols[0].replace("_close", "")
                df = _merge_futures_with_night_session(df, odf, prefix)
            else:
                df = _merge_overnight_df(df, odf, shift_days=1)

    # === 富时A50期货（分离日盘/夜盘特征） ===
    # A50: 日盘 09:00-16:30, 夜盘 17:00-04:45 (北京时间)
    #
    # open 的含义取决于数据源：
    # - 如果 open = 日盘开盘(09:00) → night_ret 精确反映隔夜涨跌 ✓
    # - 如果 open = 夜盘开盘(前日17:00) → night_ret ≈ 0% → 模型自动忽略
    # - data_fetcher.py 中的 yfinance 小时线会尝试获取真实日盘开盘价
    #
    # _merge_futures_with_night_session 确保两种情况都能正确处理
    if a50_df is not None and not a50_df.empty:
        from src.data_fetcher import _detect_open_convention
        conv = _detect_open_convention(a50_df, "a50_open", "a50_close")
        if conv == "night_session_open":
            print(f"  [A50] open = 夜盘开盘价 → night_ret 将接近0，依赖 a50_ret 和其他产品")
        elif conv == "day_session_open":
            print(f"  [A50] open = 日盘开盘价 → night_ret 可精确衡量隔夜涨跌")
        df = _merge_futures_with_night_session(df, a50_df, "a50")

    # === 境外商品期货 ===
    # 商品期货(WTI/布伦特/铜)在CME/ICE以近24小时电子盘交易，
    # open/close的"夜盘"含义不如A50/恒指清晰，但分离处理仍更安全
    if commodity_dfs:
        for odf in commodity_dfs:
            close_cols = [c for c in odf.columns if c.endswith("_close")]
            if close_cols:
                prefix = close_cols[0].replace("_close", "")
                df = _merge_futures_with_night_session(df, odf, prefix)
            else:
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
