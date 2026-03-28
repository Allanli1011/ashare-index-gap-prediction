# A股指数开盘跳空预测

预测A股主要指数（沪深300、中证500、中证1000）每日开盘价相对于前一天收盘价的高开/低开幅度。使用 LightGBM 模型 + 滚动窗口 (Walk-Forward) 回测。

## 项目结构

```
├── main.py                # 主程序入口：数据获取 → 特征工程 → 回测 → 可视化
├── src/
│   ├── data_fetcher.py    # 数据获取（akshare 免费 API）
│   ├── data_generator.py  # 模拟数据生成（API 不可用时的回退方案）
│   ├── features.py        # 特征工程 + 时间对齐
│   ├── model.py           # LightGBM 模型封装
│   └── backtest.py        # 滚动窗口回测 + 评估指标
├── output/                # 回测结果（CSV + 图表）
└── requirements.txt
```

## 快速开始

### 1. 全量回测 (Backtesting)
```bash
pip install -r requirements.txt
python main.py
```
这会运行所有历史数据，并在 `output/` 文件夹下生成回测曲线与特征重要性报告。如果 akshare API 不可用（网络限制等），程序会自动回退到模拟数据演示模式。

### 2. 实盘每日开盘预测 (Live Daily Prediction)
如果您希望模型利用最极限、最晚的数据进行推理，**强烈推荐在每个交易日早晨的 09:22 - 09:25 之间运行本脚本**。
- **09:00** 前：隔夜美股、汇率、商品、A50夜盘等已经全部收盘落定。
- **09:00** 准点：富时 A50 期指生成开盘价。
- **09:20** 准点：港股现货市场（恒生指数、恒生科技）完成开市前对盘，产生当日最新开盘价，这些是全市场离 A股开盘最近的指标。

在卡住这个时间点（如 09:23）运行以下命令，系统可以无缝且无延迟地将以上所有最新盘口读入模型：
```bash
python predict_today.py
```
**自动化运行（Crontab 示例）**：
如果你希望在 Mac/Linux 主机上实现自动运行，可以打开终端输入 `crontab -e` 并在末尾加入如下配置（每个工作日早上 09:23 自动运行，并记录输出）：
```bash
# 每天 09:23 运行并输出日志到 predict.log
23 9 * * 1-5 cd /你的/项目/路径 && /你的/Python/路径/python predict_today.py >> predict.log 2>&1
```

## 模型

- **算法**: LightGBM (Gradient Boosting Decision Tree)
- **目标**: 回归预测跳空幅度 (%)，即 `gap = (open_T - close_{T-1}) / close_{T-1} × 100`
- **回测方式**: 滚动窗口 (Walk-Forward)，初始训练窗口 500 天，每 20 天重新训练
- **验证集**: 训练窗口末尾 15% 作为 early stopping 验证集
- **评估指标**: MAE、RMSE、方向准确率、Rank IC、ICIR

---

## 特征体系（115 个特征）

### 时间对齐原则

所有特征严格遵循「**T日 9:30 A股开盘前可用**」原则，杜绝未来信息泄露：

```
时间线（北京时间）                     可用于预测 A股 T日开盘跳空？
─────────────────────────────────────────────────────────────
T-1日 15:00  A股收盘                   ✓ T-1日技术指标在此确定
T-1日 16:00  港股日盘收盘              ✓ 港股日盘数据
T-1日 16:30  A50期货日盘收盘           ✓ A50日盘数据
T-1日 17:00  A50夜盘开盘               │
T-1日 17:15  港股期货夜盘开盘          │ 夜盘交易中...
T-1日 21:30  美股开盘                  │
T-1日 23:30  欧洲市场收盘              │
T日   01:00  港股期货夜盘收盘          │
T日   03:00  A50夜盘收盘(T+1 session)  │
T日   04:00  美股收盘                  ✓ 美股/VIX/ETF/商品/美债数据
T日   04:45  A50 T+1 session 收盘      ✓ 通过 open[T]/close[T-1] 反推夜盘
T日   09:00  A50日盘开盘 → open[T]     ✓ ≈ 夜盘收盘价（间隔极短）
T日   09:15  央行公布 USD/CNY 中间价   ？ 中行汇买价时间不确定→保守不用
T日   09:30  ★ A股开盘 ← 预测时点     ← 此前所有数据均可使用
T日   11:30  SHIBOR 发布               ✗ 开盘后才公布，不可用于T日
T日   15:00  A股收盘                   ✗ 当日收盘数据用于T+1日预测
```

### 特征分类与时间处理方式

#### 一、A股自身技术指标（27个）— `shift(1)` 处理

这些指标使用 T日 OHLCV 计算，但 T日 数据在 15:00 收盘后才确定，因此统一 `shift(1)` 使用 T-1日值。

| 特征 | 说明 |
|------|------|
| `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d` | 多周期收益率 |
| `volatility_5d`, `volatility_10d`, `volatility_20d` | 滚动波动率（日收益率标准差） |
| `ma_bias_5d`, `ma_bias_10d`, `ma_bias_20d`, `ma_bias_60d` | 收盘价相对均线的偏离度 (%) |
| `vol_ratio_5d`, `vol_ratio_10d` | 成交量相对滚动均值的比率 |
| `amt_ratio_5d` | 成交额相对 5 日均值的比率 |
| `amplitude` | 振幅 (high-low)/prev_close |
| `upper_shadow`, `lower_shadow` | 上/下影线占比 |
| `intraday_ret` | 日内收益率 (close-open)/open |
| `prev_gap` | 前一日跳空幅度 |
| `turnover_ma5`, `turnover_ratio` | 换手率 5 日均值及相对比率 |
| `rsi_14` | 14 日 RSI |
| `macd`, `macd_signal`, `macd_hist` | MACD 指标三线 |
| `bb_position` | 布林带位置（当前价在上下轨之间的相对位置） |

#### 二、日历效应特征（7个）— 无需 shift

T日的日历属性在开盘前天然已知。

| 特征 | 说明 |
|------|------|
| `weekday` | 星期几（0=周一, 4=周五） |
| `month` | 月份（1-12） |
| `quarter` | 季度（1-4） |
| `is_month_start` | 是否月初（日期 ≤ 3 号） |
| `is_month_end` | 是否月末（日期 ≥ 25 号） |
| `days_since_prev` | 距上一交易日的自然日数 |
| `is_after_holiday` | 是否长假后首日（间隔 > 3 天） |

#### 三、美股指数（9个）— `date += 1 day`

美股日历日 d 收盘于北京时间 d+1 凌晨 04:00，A股 d+1 日 9:30 开盘前可用。

| 特征 | 数据源 | 说明 |
|------|--------|------|
| `us_SPX_close`, `us_SPX_open`, `us_SPX_ret` | 标普 500 | 收盘价、开盘价、日涨跌幅 |
| `us_NDX_close`, `us_NDX_open`, `us_NDX_ret` | 纳斯达克 100 | 同上 |
| `us_DJI_close`, `us_DJI_open`, `us_DJI_ret` | 道琼斯工业 | 同上 |

#### 四、美股中国相关 ETF（12个）— `date += 1 day`

美股时段交易（北京时间 21:30-04:00），直接反映海外资金对A股/中概股的隔夜定价。

| 特征 | 数据源 | 说明 |
|------|--------|------|
| `etf_ASHR_close`, `etf_ASHR_open`, `etf_ASHR_ret` | Xtrackers 沪深300 ETF | **直接挂钩沪深300** |
| `etf_FXI_close`, `etf_FXI_open`, `etf_FXI_ret` | iShares 中国大盘 ETF | 大型中概股/H股 |
| `etf_KWEB_close`, `etf_KWEB_open`, `etf_KWEB_ret` | KraneShares 中国互联网 ETF | 中概互联网板块 |
| `etf_MCHI_close`, `etf_MCHI_open`, `etf_MCHI_ret` | iShares MSCI中国 ETF | MSCI中国指数 |

#### 五、港股期货（15个）— `date += 1 day` + 夜盘反推

日盘 T日 16:00 收盘；夜盘至 T+1日 01:00。日线数据仅含日盘，**通过 `open[d+1]/close[d]-1` 反推夜盘涨跌幅**。

| 特征 | 数据源 | 说明 |
|------|--------|------|
| `hk_HSI_close`, `hk_HSI_open`, `hk_HSI_ret` | 恒生指数 | 日盘收盘/开盘/涨跌幅 |
| `hk_HSI_day_ret`, `hk_HSI_night_ret` | 恒生指数 | 日盘内涨跌、**夜盘涨跌幅** |
| `hk_HSTECH_close/open/ret/day_ret/night_ret` | 恒生科技指数 | 同上 |
| `hk_HSCEI_close/open/ret/day_ret/night_ret` | 恒生国企指数(H股) | 同上 |

#### 六、富时A50期货（6个）— `date += 1 day` + 夜盘反推

新加坡交易所，A股最直接的隔夜参考。日盘 09:00-16:30，**夜盘 17:00-04:45 横跨整个美股交易时段**。

| 特征 | 说明 |
|------|------|
| `a50_close`, `a50_open` | 日盘收盘/开盘价 |
| `a50_ret` | 日盘涨跌幅 |
| `a50_day_ret` | 日盘内收益率 (close/open-1) |
| `a50_night_ret` | **★ 夜盘涨跌幅 (open[d+1]/close[d]-1)，最有价值的隔夜信号** |

> **夜盘反推原理**: A50 次日日盘 09:00 开盘价 ≈ 前夜 04:45 夜盘收盘价（两个时段间隔仅 4 小时 15 分钟，期间无交易），因此 `open[T]/close[T-1] - 1` 即为夜盘涨跌幅。该数据在 A股 9:30 开盘前已确定。

#### 七、大宗商品期货（15个）— `date += 1 day` + 夜盘反推

以美国时间结算，同样通过 open/close 反推夜盘。

| 特征 | 数据源 | 说明 |
|------|--------|------|
| `cmd_CL_close/open/ret/day_ret/night_ret` | WTI 原油 | 全球能源风险偏好 |
| `cmd_OIL_close/open/ret/day_ret/night_ret` | 布伦特原油 | 同上 |
| `cmd_HG_close/open/ret/day_ret/night_ret` | COMEX 铜 | 全球经济景气度指标 |

#### 八、欧洲指数（9个）— `date += 1 day`

欧洲市场收盘于北京时间 d+1 ~00:30-01:00，早于美股收盘。

| 特征 | 数据源 | 说明 |
|------|--------|------|
| `eu_STOXX50_close`, `eu_STOXX50_open`, `eu_STOXX50_ret` | 欧洲斯托克 50 | 欧元区蓝筹 |
| `eu_DAX_close`, `eu_DAX_open`, `eu_DAX_ret` | 德国 DAX | 欧洲最大经济体 |
| `eu_FTSE_close`, `eu_FTSE_open`, `eu_FTSE_ret` | 英国富时 100 | 英国蓝筹 |

#### 九、VIX 恐慌指数（3个）— `date += 1 day`

| 特征 | 说明 |
|------|------|
| `vix_close` / `vix_level` | VIX 绝对水平（高VIX = 高恐慌） |
| `vix_ret` | VIX 日变化率 |

#### 十、美债收益率（5个）— `date += 1 day`

| 特征 | 说明 |
|------|------|
| `us10y_yield`, `us10y_yield_chg` | 10年期国债收益率及日变化 |
| `us2y_yield` | 2年期国债收益率 |
| `us_term_spread`, `term_spread_chg` | 期限利差 (10Y-2Y) 及日变化，反转预示衰退 |

#### 十一、汇率（5个）— 境内 `date += 1 day`

| 特征 | 数据源 | 时间处理 | 说明 |
|------|--------|----------|------|
| `usd_cny`, `usd_cny_ret` | 中行汇买价（在岸） | shift+1（T日更新时间不确定） | 在岸人民币 |
| `usd_cnh`, `usd_cnh_ret` | 离岸人民币 CNH | shift+1（日结算按美国时间） | 24h交易，更市场化 |
| `cnh_cny_spread` | 计算值 | — | **离岸-在岸价差**，反映资本流动压力 |

#### 十二、境内利率与商品（2个）— `date += 1 day`

| 特征 | 数据源 | 时间处理 | 说明 |
|------|--------|----------|------|
| `gold_price`, `gold_ret` | 上海金基准价 | shift+1（T日交易时段产生） | 黄金避险需求 |
| `shibor_on` | 银行间同业拆放利率 | shift+1（T日 11:30 才发布） | 流动性松紧 |

---

## 数据来源

所有数据通过 [AKShare](https://github.com/akfamily/akshare) 免费获取，无需付费数据订阅。

| 类别 | 数据源 | AKShare 接口 |
|------|--------|-------------|
| A股指数 | 东方财富 | `index_zh_a_hist()` |
| 美股指数 | 新浪财经 | `index_us_stock_sina()` |
| 港股指数 | 新浪财经 | `stock_hk_index_daily_sina()` |
| A50/商品期货 | 外盘期货 | `futures_foreign_hist()` |
| 美股ETF | 新浪财经 | `stock_us_daily()` |
| 黄金 | 上海金交所 | `spot_golden_benchmark_sge()` |
| 汇率 | 中国银行 | `currency_boc_safe()` |
| 美债收益率 | 东方财富 | `bond_zh_us_rate()` |
| SHIBOR | 银行间市场 | `rate_interbank()` |

## 输出文件

运行 `python main.py` 后在 `output/` 目录生成：

| 文件 | 说明 |
|------|------|
| `{指数名}_predictions.csv` | 每日预测值 vs 实际值 |
| `{指数名}_backtest.png` | 回测曲线（预测vs实际、滚动MAE、方向准确率、IC） |
| `{指数名}_scatter.png` | 预测 vs 实际散点图 |
| `{指数名}_feature_importance.png` | Top 20 特征重要性 |
| `summary_comparison.png` | 三大指数对比图 |
| `summary_metrics.csv` | 汇总评估指标 |

## 评估指标说明

| 指标 | 含义 | 参考基准 |
|------|------|----------|
| **MAE** | 平均绝对误差 (%) | 越小越好，通常 0.3-0.5% |
| **RMSE** | 均方根误差 (%) | 越小越好 |
| **方向准确率** | 预测高开/低开方向的正确率 | >50% 优于随机，>55% 有实用价值 |
| **Rank IC** | 预测值与实际值的秩相关系数 | >0.05 有信号，>0.1 较强 |
| **ICIR** | IC均值/IC标准差 | >0.5 可用，>1.0 优秀 |
