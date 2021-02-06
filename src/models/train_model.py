import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

parent_dir = str(Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)


class AveragingModels(
    BaseEstimator,
    RegressorMixin,
    TransformerMixin
):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack(
            [model.predict(X) for model in self.models_]
        )
        return np.mean(predictions, axis=1)


class StackingAveragedModels(
    BaseEstimator,
    RegressorMixin,
    TransformerMixin
):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # 元のモデルのクローンにデータを学習させる
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # クローンされたモデルを学習してからout-of-fold予測を作成
        # クローンしたメタモデルを学習させるのに必要
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # out-of-hold予測を新しい特徴として用いてクローンしたメタモデルを学習
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # 「テストデータに対するベースモデルの予測の平均」を
    # メタモデルの特徴として，最終的な予測を行う
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([
                model.predict(X) for model in base_models
            ]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def rmsle_cv(
    train: pd.DataFrame,
    y_train: pd.DataFrame,
    model: any,
    n_folds: int = 5
) -> np.ndarray:
    kf = KFold(
        n_folds,
        shuffle=True,
        random_state=42
    ).get_n_splits(train.values)
    rmse = np.sqrt(
        -cross_val_score(
            model,
            train.values,
            y_train,
            scoring='neg_mean_squared_error',
            cv=kf
        )
    )
    return (rmse)


def train_model_pipe(
    train: pd.DataFrame,
    y_train: np.ndarray,
    test: pd.DataFrame,
    run_average_base_models: bool = False,
    run_stacked_average: bool = False
) -> np.ndarray:
    # LASSO Regression
    # TODO:LASSOを復習
    lasso = make_pipeline(
        RobustScaler(),
        Lasso(alpha=0.0005, random_state=1)
    )

    # Elastic Net Regression
    # TODO:ENetを勉強
    ENet = make_pipeline(
        RobustScaler(),
        ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
    )

    # Kernel Ridge Regression
    # TODO:Kernel Ridge Regressionを勉強
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

    # Gradient Boosting Regression
    # TODO:huber lossだとなんで外れ値に頑健なの？
    GBoost = GradientBoostingRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=4,
        max_features='sqrt',
        min_samples_leaf=15,
        min_samples_split=10,
        loss='huber',  # 外れ値に頑健
        random_state=5
    )

    # XGBoost
    # パラメータの値が恣意的すぎない？
    # TODO: reg_alphaとreg_lambdaってなんだ…
    model_xgb = xgb.XGBRegressor(
        colsample_bytree=0.4603,
        gamma=0.0468,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=1.7817,
        n_estimators=2200,
        reg_alpha=0.4640,
        reg_lambda=0.8571,
        subsample=0.5213,
        random_state=7,
        nthread=-1
    )

    # LightGBM
    # TODO: 論文
    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=5,
        learning_rate=0.05,
        n_estimators=720,
        max_bin=55,
        bagging_fraction=0.8,
        bagging_freq=5,
        feature_fraction=0.2319,
        feature_fraction_seed=9,
        bagging_seed=9,
        min_data_in_leaf=6,
        min_sum_hessian_in_leaf=11
    )

    run_model = None
    if run_average_base_models:
        run_model = AveragingModels(models=(ENet, GBoost, KRR, lasso))

    if run_stacked_average:
        # 0.1081 (0.0073)
        run_model = StackingAveragedModels(
            base_models=(ENet, GBoost, KRR),
            meta_model=lasso
        )

    if run_model is not None:
        score = rmsle_cv(
            train,
            y_train,
            run_model
        )
    else:
        stacked_averaged_models = StackingAveragedModels(
            base_models=(ENet, GBoost, KRR),
            meta_model=lasso
        )
        stacked_averaged_models.fit(train.values, y_train)
        stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

        model_xgb.fit(train, y_train)
        xgb_pred = np.expm1(model_xgb.predict(test))

        model_lgb.fit(train, y_train)
        lgb_pred = np.expm1(model_lgb.predict(test.values))

        score = stacked_pred*0.70+xgb_pred*0.15+lgb_pred*0.15

    return score


if __name__ == '__main__':
    from features.build_features import clean_for_reg
    raw_d = Path('../../data/raw')
    repo_d = Path('../../reports')
    repo_fig_d = Path('../../reports/figures')

    run_average_base_models = False

    df_train = pd.read_csv(raw_d / 'train.csv')
    df_test = pd.read_csv(raw_d / 'test.csv')
    test_ID = df_test['Id']

    if run_average_base_models:
        df_train, y_train, df_test = clean_for_reg(df_train, df_test)
        score = train_model_pipe(
            df_train,
            y_train,
            df_test,
            run_average_base_models=True
        )
        print(
            'Averaged base models score:'
            f'{score.mean():.4f}({score.std():.4f})'
        )
    else:
        df_train, y_train, df_test = clean_for_reg(df_train, df_test)
        score = train_model_pipe(
            df_train,
            y_train,
            df_test,
            run_stacked_average=True
        )
        print(
            'Stacking Averaged models score:'
            f'{score.mean():.4f}({score.std():.4f})'
        )
