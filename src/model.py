"""
模型模块：LightGBM 回归模型用于预测开盘跳空幅度。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit


DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "random_state": 42,
}


class GapPredictor:
    """开盘跳空幅度预测模型。"""

    def __init__(self, params: dict | None = None):
        self.params = params or DEFAULT_PARAMS.copy()
        self.model = None
        self.feature_names = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "GapPredictor":
        """训练模型。"""
        self.feature_names = list(X_train.columns)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        model_params = self.params.copy()
        early_stopping = model_params.pop("early_stopping_rounds", 50)
        random_state = model_params.pop("random_state", 42)
        n_estimators = model_params.pop("n_estimators", 500)

        callbacks = [lgb.early_stopping(early_stopping, verbose=False)]

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            **model_params,
        )
        self.model.fit(
            X_train, y_train,
            callbacks=callbacks if X_val is not None else None,
            **fit_params,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测。"""
        return self.model.predict(X)

    def feature_importance(self) -> pd.DataFrame:
        """获取特征重要性。"""
        imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return imp

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict:
        """时间序列交叉验证。"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {"mae": [], "rmse": [], "direction_acc": [], "ic": []}

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            self.train(X_tr, y_tr, X_va, y_va)
            preds = self.predict(X_va)

            scores["mae"].append(np.mean(np.abs(preds - y_va)))
            scores["rmse"].append(np.sqrt(np.mean((preds - y_va) ** 2)))
            scores["direction_acc"].append(
                np.mean(np.sign(preds) == np.sign(y_va))
            )
            corr = np.corrcoef(preds, y_va)[0, 1]
            scores["ic"].append(corr if not np.isnan(corr) else 0)

        return {k: np.mean(v) for k, v in scores.items()}
