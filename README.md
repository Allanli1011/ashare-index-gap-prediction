# A股指数开盘跳空预测

预测A股主要指数（沪深300、中证500、中证1000）每日开盘价相对于前一天收盘价的高开/低开幅度。

## 项目结构

```
├── main.py              # 主程序入口
├── src/
│   ├── data_fetcher.py  # 数据获取（akshare）
│   ├── features.py      # 特征工程
│   ├── model.py         # LightGBM 模型
│   └── backtest.py      # 滚动窗口回测
├── output/              # 回测结果和图表
└── requirements.txt     # 依赖
```

## 特征体系

- **技术指标**: 多周期收益率、波动率、均线偏离度、RSI、MACD、布林带、成交量比率等
- **隔夜因子**: 美股三大指数（标普500、纳斯达克、道琼斯）涨跌幅、黄金价格、美元兑人民币汇率
- **日历效应**: 星期几、月份、节假日效应、季度
- **市场微观**: 换手率、振幅、上下影线

## 使用方法

```bash
pip install -r requirements.txt
python main.py
```

## 模型

- **算法**: LightGBM (Gradient Boosting Decision Tree)
- **目标**: 回归预测跳空幅度 (%)
- **回测方式**: 滚动窗口 (Walk-Forward)，初始训练500天，每20天重训练
- **评估指标**: MAE、RMSE、方向准确率、Rank IC、ICIR

## 数据来源

- [AKShare](https://github.com/akfamily/akshare) - 免费开源的金融数据接口
