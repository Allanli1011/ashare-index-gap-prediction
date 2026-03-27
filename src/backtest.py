"""
回测模块：滚动窗口（Walk-Forward）回测。
"""
import numpy as np
import pandas as pd
from .model import GapPredictor


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "gap_pct",
    initial_train_days: int = 500,
    retrain_every: int = 20,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    """滚动窗口回测。

    Args:
        df: 含特征和目标的DataFrame（已按日期排序，无NaN）。
        feature_cols: 特征列名。
        target_col: 目标列名。
        initial_train_days: 初始训练窗口大小。
        retrain_every: 每隔多少天重新训练。
        val_ratio: 验证集占训练集的比例。

    Returns:
        包含日期、真实值、预测值的DataFrame。
    """
    results = []
    n = len(df)
    model = None
    last_train_idx = -retrain_every  # 强制第一次训练

    for i in range(initial_train_days, n):
        # 是否需要重新训练
        if i - last_train_idx >= retrain_every or model is None:
            train_data = df.iloc[:i]
            X_all = train_data[feature_cols]
            y_all = train_data[target_col]

            # 划分训练/验证
            val_size = max(int(len(X_all) * val_ratio), 20)
            X_train = X_all.iloc[:-val_size]
            y_train = y_all.iloc[:-val_size]
            X_val = X_all.iloc[-val_size:]
            y_val = y_all.iloc[-val_size:]

            model = GapPredictor()
            model.train(X_train, y_train, X_val, y_val)
            last_train_idx = i

        # 预测
        X_test = df.iloc[[i]][feature_cols]
        pred = model.predict(X_test)[0]
        actual = df.iloc[i][target_col]

        results.append({
            "date": df.iloc[i]["date"],
            "actual": actual,
            "predicted": pred,
        })

    return pd.DataFrame(results)


def compute_metrics(results: pd.DataFrame) -> dict:
    """计算回测指标。"""
    actual = results["actual"].values
    predicted = results["predicted"].values

    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    direction_acc = np.mean(np.sign(predicted) == np.sign(actual))

    # IC (Information Coefficient) - 滚动20日rank IC
    ic_list = []
    window = 20
    for i in range(window, len(results)):
        a = actual[i - window:i]
        p = predicted[i - window:i]
        from scipy.stats import spearmanr
        corr, _ = spearmanr(a, p)
        if not np.isnan(corr):
            ic_list.append(corr)

    ic_mean = np.mean(ic_list) if ic_list else 0
    ic_std = np.std(ic_list) if ic_list else 1
    icir = ic_mean / ic_std if ic_std > 0 else 0

    # 分组统计：高开 vs 低开
    high_open_mask = actual > 0
    low_open_mask = actual < 0

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "方向准确率": direction_acc,
        "IC均值": ic_mean,
        "ICIR": icir,
        "样本数": len(results),
        "高开样本数": int(high_open_mask.sum()),
        "低开样本数": int(low_open_mask.sum()),
    }

    # 分档准确率
    for threshold in [0.0, 0.1, 0.3, 0.5]:
        if threshold == 0.0:
            label = "方向准确率(全样本)"
            mask = np.ones(len(actual), dtype=bool)
        else:
            label = f"方向准确率(|gap|>{threshold}%)"
            mask = np.abs(actual) > threshold

        if mask.sum() > 0:
            metrics[label] = np.mean(
                np.sign(predicted[mask]) == np.sign(actual[mask])
            )

    return metrics


def compute_daily_metrics_rolling(results: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """计算滚动窗口指标用于绘图。"""
    out = results.copy()
    out["abs_error"] = np.abs(out["predicted"] - out["actual"])
    out["direction_correct"] = (np.sign(out["predicted"]) == np.sign(out["actual"])).astype(int)

    out["rolling_mae"] = out["abs_error"].rolling(window).mean()
    out["rolling_dir_acc"] = out["direction_correct"].rolling(window).mean()

    # 滚动IC
    ic_values = []
    for i in range(len(out)):
        if i < window:
            ic_values.append(np.nan)
        else:
            from scipy.stats import spearmanr
            a = out["actual"].iloc[i - window:i].values
            p = out["predicted"].iloc[i - window:i].values
            corr, _ = spearmanr(a, p)
            ic_values.append(corr)
    out["rolling_ic"] = ic_values

    return out
