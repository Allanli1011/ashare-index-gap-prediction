"""
A股指数开盘跳空预测 - 主程序

预测沪深300、中证500、中证1000每日开盘价相对于前一天收盘价的跳空幅度。
使用 LightGBM 模型 + 滚动窗口回测。
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_fetcher import fetch_all_data, INDEX_CODES
from src.data_generator import generate_all_data
from src.features import build_features, get_feature_columns
from src.backtest import walk_forward_backtest, compute_metrics, compute_daily_metrics_rolling
from src.model import GapPredictor

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "output"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_backtest_results(results: pd.DataFrame, index_name: str):
    """绘制回测结果图表。"""
    rolling = compute_daily_metrics_rolling(results, window=60)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"{index_name} - Opening Gap Prediction Backtest", fontsize=16, y=0.98)

    # 1. 预测 vs 实际
    ax = axes[0]
    ax.plot(rolling["date"], rolling["actual"], alpha=0.5, linewidth=0.8, label="Actual Gap %")
    ax.plot(rolling["date"], rolling["predicted"], alpha=0.5, linewidth=0.8, label="Predicted Gap %")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_ylabel("Gap %")
    ax.set_title("Actual vs Predicted Opening Gap")
    ax.legend()

    # 2. 滚动MAE
    ax = axes[1]
    ax.plot(rolling["date"], rolling["rolling_mae"], color="red", linewidth=1)
    ax.set_ylabel("MAE %")
    ax.set_title("Rolling 60-day MAE")

    # 3. 滚动方向准确率
    ax = axes[2]
    ax.plot(rolling["date"], rolling["rolling_dir_acc"], color="green", linewidth=1)
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="50% baseline")
    ax.set_ylabel("Accuracy")
    ax.set_title("Rolling 60-day Direction Accuracy")
    ax.legend()

    # 4. 滚动IC
    ax = axes[3]
    ax.plot(rolling["date"], rolling["rolling_ic"], color="blue", linewidth=1)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_ylabel("Rank IC")
    ax.set_title("Rolling 60-day Rank IC")
    ax.set_xlabel("Date")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{index_name}_backtest.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表已保存] {path}")


def plot_scatter(results: pd.DataFrame, index_name: str):
    """绘制预测 vs 实际散点图。"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(results["actual"], results["predicted"], alpha=0.3, s=10)
    lims = [
        min(results["actual"].min(), results["predicted"].min()),
        max(results["actual"].max(), results["predicted"].max()),
    ]
    ax.plot(lims, lims, "r--", alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Actual Gap %")
    ax.set_ylabel("Predicted Gap %")
    ax.set_title(f"{index_name} - Predicted vs Actual Opening Gap")
    ax.legend()
    ax.set_aspect("equal")

    path = os.path.join(OUTPUT_DIR, f"{index_name}_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表已保存] {path}")


def plot_feature_importance(model: GapPredictor, index_name: str, top_n: int = 20):
    """绘制特征重要性。"""
    imp = model.feature_importance().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(imp)), imp["importance"].values, align="center")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"{index_name} - Top {top_n} Feature Importance")

    path = os.path.join(OUTPUT_DIR, f"{index_name}_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表已保存] {path}")


def plot_summary_comparison(all_metrics: dict):
    """绘制多指数对比图。"""
    names = list(all_metrics.keys())
    mae_vals = [all_metrics[n]["MAE"] for n in names]
    dir_vals = [all_metrics[n]["方向准确率"] for n in names]
    ic_vals = [all_metrics[n]["IC均值"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(names, mae_vals, color=["#2196F3", "#FF9800", "#4CAF50"])
    axes[0].set_title("MAE Comparison")
    axes[0].set_ylabel("MAE %")

    axes[1].bar(names, dir_vals, color=["#2196F3", "#FF9800", "#4CAF50"])
    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("Direction Accuracy Comparison")
    axes[1].set_ylabel("Accuracy")

    axes[2].bar(names, ic_vals, color=["#2196F3", "#FF9800", "#4CAF50"])
    axes[2].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_title("Mean IC Comparison")
    axes[2].set_ylabel("IC")

    plt.suptitle("Multi-Index Backtest Comparison", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summary_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表已保存] {path}")


def run_pipeline():
    """运行完整的预测回测流程。"""
    ensure_output_dir()

    # ==================== 1. 数据获取 ====================
    print("=" * 60)
    print("步骤1: 获取数据")
    print("=" * 60)
    try:
        data = fetch_all_data(start_date="20180101")
    except Exception as e:
        print(f"  [警告] akshare 数据获取失败: {e}")
        print("  [回退] 使用模拟数据进行演示 ...")
        data = generate_all_data()

    all_metrics = {}
    all_results = {}

    # ==================== 2. 逐指数处理 ====================
    for index_name in INDEX_CODES:
        print("\n" + "=" * 60)
        print(f"处理指数: {index_name}")
        print("=" * 60)

        index_df = data[index_name]
        print(f"  原始数据: {len(index_df)} 条记录, "
              f"日期范围: {index_df['date'].min().date()} ~ {index_df['date'].max().date()}")

        # ---- 特征工程 ----
        print("  构建特征 ...")
        df = build_features(
            index_df,
            overseas_dfs=data.get("overseas", []),
            gold_df=data.get("gold", pd.DataFrame()),
            usd_cny_df=data.get("usd_cny", pd.DataFrame()),
            shibor_df=data.get("shibor", pd.DataFrame()),
            hk_dfs=data.get("hk", []),
            a50_df=data.get("a50", pd.DataFrame()),
            commodity_dfs=data.get("commodities", []),
            vix_df=data.get("vix", pd.DataFrame()),
            us_treasury_df=data.get("us_treasury", pd.DataFrame()),
            china_etf_dfs=data.get("china_etfs", []),
            euro_dfs=data.get("euro", []),
            usd_cnh_df=data.get("usd_cnh", pd.DataFrame()),
        )

        feature_cols = get_feature_columns(df)
        print(f"  特征数量: {len(feature_cols)}")

        # 删除含 NaN 的行
        df_clean = df.dropna(subset=feature_cols + ["gap_pct"]).reset_index(drop=True)
        print(f"  清洗后数据: {len(df_clean)} 条")

        # ---- 回测 ----
        print("  开始滚动窗口回测 ...")
        results = walk_forward_backtest(
            df_clean,
            feature_cols,
            target_col="gap_pct",
            initial_train_days=500,
            retrain_every=20,
        )
        print(f"  回测完成: {len(results)} 个预测样本")

        # ---- 指标 ----
        metrics = compute_metrics(results)
        all_metrics[index_name] = metrics
        all_results[index_name] = results

        print(f"\n  === {index_name} 回测指标 ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

        # ---- 绘图 ----
        print("  生成图表 ...")
        plot_backtest_results(results, index_name)
        plot_scatter(results, index_name)

        # 训练最终模型以获取特征重要性
        final_model = GapPredictor()
        X_all = df_clean[feature_cols]
        y_all = df_clean["gap_pct"]
        val_size = int(len(X_all) * 0.15)
        final_model.train(
            X_all.iloc[:-val_size], y_all.iloc[:-val_size],
            X_all.iloc[-val_size:], y_all.iloc[-val_size:],
        )
        plot_feature_importance(final_model, index_name)

        # 保存预测结果
        results.to_csv(os.path.join(OUTPUT_DIR, f"{index_name}_predictions.csv"), index=False)

    # ==================== 3. 汇总对比 ====================
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)

    plot_summary_comparison(all_metrics)

    # 保存汇总指标
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"))
    print("\n汇总指标:")
    print(metrics_df.to_string())

    print(f"\n所有结果已保存到 {OUTPUT_DIR}/ 目录")
    return all_metrics, all_results


if __name__ == "__main__":
    run_pipeline()
