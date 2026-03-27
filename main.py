"""
A股指数开盘跳空预测 - 主程序

预测沪深300、中证500、中证1000每日开盘价相对于前一天收盘价的跳空幅度。
使用 LightGBM 模型 + 滚动窗口回测。
"""
import os
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_fetcher import fetch_all_data, INDEX_CODES
from src.data_generator import generate_all_data
from src.features import build_features, get_feature_columns
from src.backtest import (
    walk_forward_backtest, compute_metrics,
    compute_daily_metrics_rolling, summarize_feature_importance,
)
from src.model import GapPredictor

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "output"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 可视化函数
# ============================================================

def plot_backtest_results(results: pd.DataFrame, index_name: str):
    """绘制回测结果图表。"""
    rolling = compute_daily_metrics_rolling(results, window=60)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"{index_name} - Opening Gap Prediction Backtest", fontsize=16, y=0.98)

    ax = axes[0]
    ax.plot(rolling["date"], rolling["actual"], alpha=0.5, linewidth=0.8, label="Actual Gap %")
    ax.plot(rolling["date"], rolling["predicted"], alpha=0.5, linewidth=0.8, label="Predicted Gap %")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_ylabel("Gap %")
    ax.set_title("Actual vs Predicted Opening Gap")
    ax.legend()

    ax = axes[1]
    ax.plot(rolling["date"], rolling["rolling_mae"], color="red", linewidth=1)
    ax.set_ylabel("MAE %")
    ax.set_title("Rolling 60-day MAE")

    ax = axes[2]
    ax.plot(rolling["date"], rolling["rolling_dir_acc"], color="green", linewidth=1)
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="50% baseline")
    ax.set_ylabel("Accuracy")
    ax.set_title("Rolling 60-day Direction Accuracy")
    ax.legend()

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


def plot_feature_importance_avg(feat_summary: pd.DataFrame, index_name: str, top_n: int = 20):
    """绘制跨窗口平均特征重要性（含稳定性指标）。"""
    if feat_summary.empty:
        return
    data = feat_summary.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(data)), data["avg_importance"].values, align="center", color="#2196F3")

    # 用颜色深浅表示 top10 稳定率
    for i, (_, row) in enumerate(data.iterrows()):
        alpha = 0.3 + 0.7 * row["top10_rate"]
        bars[i].set_alpha(alpha)

    ax.set_yticks(range(len(data)))
    labels = [f"{row['feature']}  (top10: {row['top10_rate']:.0%})" for _, row in data.iterrows()]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Average Importance (across all retraining windows)")
    ax.set_title(f"{index_name} - Feature Importance (stability = bar opacity)")

    path = os.path.join(OUTPUT_DIR, f"{index_name}_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表已保存] {path}")


def plot_feature_rank_evolution(importance_df: pd.DataFrame, index_name: str, top_n: int = 10):
    """绘制 top 特征排名随时间的变化（稳定性可视化）。"""
    if importance_df.empty:
        return

    # 找出平均排名最高的 top_n 特征
    avg_rank = importance_df.groupby("feature")["importance"].mean().nlargest(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 6))
    for feat in avg_rank:
        subset = importance_df[importance_df["feature"] == feat].sort_values("retrain_date")
        # 计算每次重训练中的排名
        ranks = []
        for rid in subset["retrain_id"].unique():
            window = importance_df[importance_df["retrain_id"] == rid].copy()
            window["rank"] = window["importance"].rank(ascending=False)
            r = window.loc[window["feature"] == feat, "rank"]
            if not r.empty:
                ranks.append({"retrain_date": window["retrain_date"].iloc[0], "rank": r.iloc[0]})
        if ranks:
            rdf = pd.DataFrame(ranks)
            ax.plot(rdf["retrain_date"], rdf["rank"], marker=".", linewidth=1.2, markersize=4, label=feat)

    ax.invert_yaxis()
    ax.set_ylabel("Rank (lower = more important)")
    ax.set_xlabel("Retraining Date")
    ax.set_title(f"{index_name} - Top {top_n} Feature Rank Evolution")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    path = os.path.join(OUTPUT_DIR, f"{index_name}_feature_rank_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表已保存] {path}")


def plot_summary_comparison(all_metrics: dict):
    """绘制多指数对比图。"""
    names = list(all_metrics.keys())
    if not names:
        return
    mae_vals = [all_metrics[n]["MAE"] for n in names]
    dir_vals = [all_metrics[n]["方向准确率"] for n in names]
    ic_vals = [all_metrics[n]["IC均值"] for n in names]

    colors = ["#2196F3", "#FF9800", "#4CAF50"][:len(names)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(names, mae_vals, color=colors)
    axes[0].set_title("MAE Comparison")
    axes[0].set_ylabel("MAE %")

    axes[1].bar(names, dir_vals, color=colors)
    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("Direction Accuracy Comparison")
    axes[1].set_ylabel("Accuracy")

    axes[2].bar(names, ic_vals, color=colors)
    axes[2].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_title("Mean IC Comparison")
    axes[2].set_ylabel("IC")

    plt.suptitle("Multi-Index Backtest Comparison", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "summary_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表已保存] {path}")


# ============================================================
# 文本报告生成
# ============================================================

def generate_report(
    all_metrics: dict,
    all_feat_summaries: dict,
    data_mode: str,
) -> str:
    """生成文本汇总报告。"""
    lines = []
    lines.append("=" * 70)
    lines.append("A股指数开盘跳空预测 — 回测报告")
    lines.append("=" * 70)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"数据来源: {'akshare (真实数据)' if data_mode == 'real' else '模拟数据 (演示模式)'}")
    lines.append("")

    # 汇总表
    lines.append("-" * 70)
    lines.append("回测指标汇总")
    lines.append("-" * 70)
    header = f"{'指数':<10} {'MAE':>8} {'RMSE':>8} {'方向准确率':>10} {'IC均值':>8} {'ICIR':>8} {'样本数':>8}"
    lines.append(header)
    lines.append("-" * 70)
    for name, m in all_metrics.items():
        row = (f"{name:<10} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
               f"{m['方向准确率']:>10.2%} {m['IC均值']:>8.4f} {m['ICIR']:>8.4f} "
               f"{m['样本数']:>8.0f}")
        lines.append(row)
    lines.append("")

    # 分档准确率
    lines.append("-" * 70)
    lines.append("分档方向准确率（跳空幅度越大，预测越准）")
    lines.append("-" * 70)
    thresholds = ["方向准确率(全样本)", "方向准确率(|gap|>0.1%)",
                  "方向准确率(|gap|>0.3%)", "方向准确率(|gap|>0.5%)"]
    header2 = f"{'指数':<10}" + "".join(f" {t.split('(')[1].rstrip(')'):>16}" for t in thresholds)
    lines.append(header2)
    for name, m in all_metrics.items():
        vals = "".join(f" {m.get(t, 0):>16.2%}" for t in thresholds)
        lines.append(f"{name:<10}{vals}")
    lines.append("")

    # 各指数 Top 特征
    for name, feat_summary in all_feat_summaries.items():
        if feat_summary.empty:
            continue
        lines.append("-" * 70)
        lines.append(f"{name} — Top 10 最重要特征（跨所有重训练窗口平均）")
        lines.append("-" * 70)
        lines.append(f"{'排名':>4} {'特征':<30} {'平均重要性':>10} {'平均排名':>8} {'Top10稳定率':>12}")
        for i, (_, row) in enumerate(feat_summary.head(10).iterrows(), 1):
            lines.append(
                f"{i:>4} {row['feature']:<30} {row['avg_importance']:>10.1f} "
                f"{row['avg_rank']:>8.1f} {row['top10_rate']:>12.0%}"
            )
        lines.append("")

    # 关键发现
    lines.append("-" * 70)
    lines.append("关键发现")
    lines.append("-" * 70)

    if all_metrics:
        best_idx = max(all_metrics, key=lambda k: all_metrics[k]["方向准确率"])
        worst_idx = min(all_metrics, key=lambda k: all_metrics[k]["方向准确率"])
        lines.append(f"- 方向准确率最高的指数: {best_idx} ({all_metrics[best_idx]['方向准确率']:.2%})")
        lines.append(f"- 方向准确率最低的指数: {worst_idx} ({all_metrics[worst_idx]['方向准确率']:.2%})")

        best_ic = max(all_metrics, key=lambda k: all_metrics[k]["ICIR"])
        lines.append(f"- ICIR 最高的指数: {best_ic} ({all_metrics[best_ic]['ICIR']:.4f})")

        for name, m in all_metrics.items():
            if m["方向准确率"] > 0.55:
                lines.append(f"- {name}: 方向准确率 {m['方向准确率']:.2%} 超过 55% 阈值，具有实用价值")
            gap_05 = m.get("方向准确率(|gap|>0.5%)", 0)
            if gap_05 > 0.6:
                lines.append(f"- {name}: 大幅跳空(>0.5%)方向准确率达 {gap_05:.2%}，信号较强")

    lines.append("")
    lines.append("=" * 70)
    lines.append("报告结束")
    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================

def process_single_index(index_name: str, data: dict, feature_cols_out: dict) -> tuple:
    """处理单个指数的完整流程。返回 (metrics, results, feat_summary) 或抛出异常。"""
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
    feature_cols_out[index_name] = feature_cols
    print(f"  特征数量: {len(feature_cols)}")

    # 删除含 NaN 的行
    df_clean = df.dropna(subset=feature_cols + ["gap_pct"]).reset_index(drop=True)
    print(f"  清洗后数据: {len(df_clean)} 条")

    # ---- 回测 ----
    print("  开始滚动窗口回测 ...")
    results, importance_df = walk_forward_backtest(
        df_clean,
        feature_cols,
        target_col="gap_pct",
        initial_train_days=500,
        retrain_every=20,
    )
    print(f"  回测完成: {len(results)} 个预测样本, "
          f"{importance_df['retrain_id'].nunique() if not importance_df.empty else 0} 次重训练")

    # ---- 指标 ----
    metrics = compute_metrics(results)

    print(f"\n  === {index_name} 回测指标 ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # ---- 特征重要性汇总 ----
    feat_summary = summarize_feature_importance(importance_df, top_n=30)

    # ---- 绘图 ----
    print("  生成图表 ...")
    plot_backtest_results(results, index_name)
    plot_scatter(results, index_name)
    plot_feature_importance_avg(feat_summary, index_name)
    plot_feature_rank_evolution(importance_df, index_name)

    # 保存预测结果和特征重要性
    results.to_csv(os.path.join(OUTPUT_DIR, f"{index_name}_predictions.csv"), index=False)
    if not feat_summary.empty:
        feat_summary.to_csv(os.path.join(OUTPUT_DIR, f"{index_name}_feature_summary.csv"), index=False)
    if not importance_df.empty:
        importance_df.to_csv(os.path.join(OUTPUT_DIR, f"{index_name}_importance_history.csv"), index=False)

    return metrics, results, feat_summary


def run_pipeline():
    """运行完整的预测回测流程。"""
    ensure_output_dir()

    # ==================== 1. 数据获取 ====================
    print("=" * 60)
    print("步骤1: 获取数据")
    print("=" * 60)
    data_mode = "real"
    try:
        data = fetch_all_data(start_date="20180101")
    except Exception as e:
        print(f"  [警告] akshare 数据获取失败: {e}")
        print("  [回退] 使用模拟数据进行演示 ...")
        data = generate_all_data()
        data_mode = "simulated"

    all_metrics = {}
    all_results = {}
    all_feat_summaries = {}
    feature_cols_out = {}

    # ==================== 2. 逐指数处理 ====================
    for index_name in INDEX_CODES:
        print("\n" + "=" * 60)
        print(f"处理指数: {index_name}")
        print("=" * 60)

        try:
            metrics, results, feat_summary = process_single_index(
                index_name, data, feature_cols_out,
            )
            all_metrics[index_name] = metrics
            all_results[index_name] = results
            all_feat_summaries[index_name] = feat_summary
        except Exception as e:
            print(f"\n  [错误] {index_name} 处理失败: {e}")
            traceback.print_exc()
            print(f"  [跳过] 继续处理下一个指数 ...")

    # ==================== 3. 汇总对比 ====================
    if all_metrics:
        print("\n" + "=" * 60)
        print("汇总对比")
        print("=" * 60)

        plot_summary_comparison(all_metrics)

        # 保存汇总指标
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"))
        print("\n汇总指标:")
        print(metrics_df.to_string())

        # ==================== 4. 生成文本报告 ====================
        report = generate_report(all_metrics, all_feat_summaries, data_mode)
        report_path = os.path.join(OUTPUT_DIR, "backtest_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[报告已保存] {report_path}")
        print("\n" + report)
    else:
        print("\n[错误] 所有指数均处理失败，无汇总结果")

    print(f"\n所有结果已保存到 {OUTPUT_DIR}/ 目录")
    return all_metrics, all_results


if __name__ == "__main__":
    run_pipeline()
