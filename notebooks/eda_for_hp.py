import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

parent_dir = str(Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)


# 欠損値確認
# 欠損値率が高いやつは消しても予測に問題なさそう
def check_missing_data(
        df: pd.DataFrame,
        head_num: int = 20,
        print_missing: bool = False
) -> pd.DataFrame:
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (
        df.isnull().sum()/df.isnull().count()
    ).sort_values(ascending=False)
    missing_data = pd.concat(
        [total, percent],
        axis=1,
        keys=['Total', 'Percent']
    )

    if print_missing:
        print(missing_data.head(head_num))
    return missing_data


# 欠損値のある列を削除
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # 欠損値削除
    miss_df = check_missing_data(df)
    df = df.drop(miss_df[miss_df['Total'] > 1].index, 1)
    df = df.dropna(how='any')

    # 外れ値削除
    df = remove_outliars(df)
    # 'TotalBsmtSF'の0の値が入ってるものにフラグを立て，dfに加える
    df = add_zero_flag(df)

    # log変換
    df['GrLivArea'] = np.log(df['GrLivArea'])
    df['SalePrice'] = np.log(df['SalePrice'])
    df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(
        df[df['TotalBsmtSF'] > 0]['TotalBsmtSF']
    )
    # カテゴリ変数をダミー変数に変換
    df = pd.get_dummies(df)
    return df


# 標準化
def stan_df(df: pd.DataFrame, tar: str = 'SalePrice'):
    saleprice_scaled = StandardScaler().fit_transform(
        df[tar].values[:, np.newaxis]
    )

    low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
    high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]

    print(low_range)
    print(high_range)


def remove_outliars(df: pd.DataFrame, check_df: bool = False) -> pd.DataFrame:
    if check_df:
        print(df.sort_values(by='GrLivArea', ascending=False)[:2])
    df = df.drop(df[df['Id'] == 1299].index)
    df = df.drop(df[df['Id'] == 524].index)
    return df


def add_zero_flag(df: pd.DataFrame):
    df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
    df['HasBsmt'] = 0
    df.loc[df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    return df


if __name__ == '__main__':
    from src.visualization.visualize import (box_plot, check_SalePrice,
                                             corr_mat, hist_norm_plot,
                                             pair_plot, prob_plot,
                                             scatter_plot, zoom_corr_mat)

    raw_d = Path('../data/raw')
    repo_fig_d = Path('../reports/figures')

    df_train = pd.read_csv(raw_d / 'train.csv')
    df_train = clean_df(df_train)

    # check_SalePrice(df_train)
    # scatter_plot(df_train, var='TotalBsmtSF')
    # box_plot(df_train, var='OverallQual')
    # box_plot(df_train, var='YearBuilt')
    # corr_mat(df_train)
    # zoom_corr_mat(df_train)

    # pair_plot(
    #     df_train,
    #     save_fig_d=repo_fig_d / 'home_price_pairplot.jpg',
    #     height=2.5
    # )

    # stan_df(df_train)

    # hist_norm_plot(
    #     df_train, save_fig_d=repo_fig_d /
    #     'home_price_hisr_norm_plot.jpg'
    # )

    # prob_plot(
    #     df_train,
    #     save_fig_d=repo_fig_d / 'home_price_prob_plot.jpg'
    # )

    prob_plot(
        df_train[df_train['TotalBsmtSF'] > 0],
        tar='TotalBsmtSF'
    )

    scatter_plot(
        df_train[df_train['TotalBsmtSF'] > 0],
        var='TotalBsmtSF',
        ymin=None,
        ymax=None,
        figsize=(12, 8)
    )
