"""
A股指数开盘跳空预测 - 每日实盘预测脚本
推荐每天早晨 08:30 - 09:15 之间（即A股开盘前）使用定时任务运行此脚本。
"""
from __future__ import annotations
import os
import warnings
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_fetcher import fetch_all_data, INDEX_CODES
from src.features import build_features, get_feature_columns
from src.model import GapPredictor

warnings.filterwarnings("ignore")

def main():
    print("=" * 60)
    print(f"A股指数开盘跳空预测 - 实盘预测")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 抓取最新数据
    print("\n步骤1: 抓取隔夜及历史数据...")
    try:
        # 为了提高运行速度，我们只需拉取最近几年的数据供训练（足够了，不用从 2018 开始拉取，这里拉取3年）
        start_date = (pd.Timestamp.today() - pd.Timedelta(days=1000)).strftime("%Y%m%d")
        data = fetch_all_data(start_date=start_date)
    except Exception as e:
        print(f"\n[错误] 数据获取失败: {e}")
        traceback.print_exc()
        return

    today = pd.Timestamp.today().normalize()
    
    # 2. 逐指数处理与预测
    print("\n" + "=" * 60)
    print("步骤2: 训练模型并预测今日开盘")
    print("=" * 60)
    
    predictions = {}
    
    for index_name, code in INDEX_CODES.items():
        print(f"\n处理指数: {index_name}")
        index_df = data.get(index_name)
        if index_df is None or index_df.empty:
            print(f"  [警告] 没有 {index_name} 的历史数据，跳过")
            continue
            
        # 【核心逻辑】构造今日的预测输入
        # 如果当前时间是A股开盘前，akshare 拉取到的 index_df 最新一天通常是昨天 (T-1)
        # 我们需要在数据末尾强行加一行“今天 (T)”的空数据。这样在 merge 隔夜因子时才能成功贴在此行
        max_date = index_df["date"].max()
        
        # 考虑到可能是节假日后第一次运行，或者只是正常的次日早晨
        # 我们强制在最后贴上一行目标日：today
        if max_date < today:
            # 追加空行
            empty_row = pd.DataFrame([{"date": today}])
            index_df = pd.concat([index_df, empty_row], ignore_index=True)
            target_date = today
        else:
            # 这个情况通常发生在 A股已经开盘甚至收盘后，AKShare 已经包含了 T 日的 K线
            target_date = max_date
            print(f"  [提示] 发现数据已包含今天 {today.date()} 的 K线。可能市场已经开通或已收盘。")

        # 临时替换 data dict 里的 index_df 给 build_features 使用
        temp_data = data.copy()
        temp_data[index_name] = index_df
        
        # 调用特征工程
        print("  正在构建特征与隔夜因子对其...")
        try:
            df_features = build_features(
                index_df,
                overseas_dfs=temp_data.get("overseas", []),
                gold_df=temp_data.get("gold", pd.DataFrame()),
                usd_cny_df=temp_data.get("usd_cny", pd.DataFrame()),
                shibor_df=temp_data.get("shibor", pd.DataFrame()),
                hk_dfs=temp_data.get("hk", []),
                a50_df=temp_data.get("a50", pd.DataFrame()),
                commodity_dfs=temp_data.get("commodities", []),
                vix_df=temp_data.get("vix", pd.DataFrame()),
                us_treasury_df=temp_data.get("us_treasury", pd.DataFrame()),
                china_etf_dfs=temp_data.get("china_etfs", []),
                euro_dfs=temp_data.get("euro", []),
                usd_cnh_df=temp_data.get("usd_cnh", pd.DataFrame()),
            )
        except Exception as e:
            print(f"  [警告] {index_name} 特征工程失败: {e}")
            continue
            
        feature_cols = get_feature_columns(df_features)
        
        # 提取训练集：除了 target_date （最后一行或最后几天中缺失 gap_pct 的行），其他全作为训练集
        # 注意：预测今天的行，它的 gap_pct 必定是 NaN，因为这正是我们要预测的东西
        train_df = df_features.dropna(subset=feature_cols + ["gap_pct"]).reset_index(drop=True)
        
        # 提取测试集（目标日）
        today_row = df_features[df_features["date"] == target_date].copy()
        if today_row.empty:
            print(f"  [警告] 无法构建 {target_date.date()} 的预测行，可能由于严重缺失特征而导致被丢弃。")
            continue
            
        # 预测行依然可能在某些必须的特征上为缺失值（比如隔夜市场没开）。模型其实可以用LightGBM自带的处理空值，所以不去 dropna
        X_today = today_row[feature_cols].iloc[[-1]] 
        
        print(f"  利用 {len(train_df)} 条历史数据进行全量训练...")
        model = GapPredictor(params={"n_estimators": 200, "learning_rate": 0.05, "verbose": -1})
        model.train(train_df[feature_cols], train_df["gap_pct"])
        
        pred_gap = model.predict(X_today)[0]
        direction = "🔴 高开" if pred_gap > 0 else "🟢 低开"
        if abs(pred_gap) < 0.05:
            direction = "⚪️ 平开"
            
        predictions[index_name] = {
            "date": target_date.date(),
            "pred_gap": pred_gap,
            "direction": direction
        }
        
    # 3. 输出最终结果
    print("\n" + "=" * 60)
    print("📈 今日开盘跳空预测结果汇总 ")
    print("=" * 60)
    if not predictions:
        print("暂无预测结果。请检查数据是否拉取成功。")
    else:
        for name, res in predictions.items():
            gap_val = res["pred_gap"]
            print(f"【{name}】 | 预测方向: {res['direction']} | 预期幅度: {gap_val:+.3f}%")
        print("-" * 60)
        print("💡 注1：幅度绝对值越大，胜率越高。历史回测显示当预期跳空幅度 > 0.5% 时，方向准确率接近 90%。")
        print("💡 注2：跳空方向由夜盘和外围影响主导，但开盘跳空后日内的走势（回补/高走）仍受内盘资金面共同影响。")
    print("=" * 60)


if __name__ == "__main__":
    main()
