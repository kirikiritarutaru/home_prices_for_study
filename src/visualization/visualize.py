from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm

sns.set()


def check_SalePrice(df, feat='SalePrice'):
    print(df[feat].describe())
    print(f'Skewness: {df[feat].skew()}')  # 歪度
    print(f'Kurtosis: {df[feat].kurt()}')  # 尖度

    # だいたい180000ぐらいに固まってて，高いものは750000ぐらいいってる
    sns.histplot(df[feat], kde=True)
    plt.show()


# scatterでtargetとvarianceの関係を確認
def scatter_plot(
        df: pd.DataFrame,
        tar: str = 'SalePrice',
        var: str = 'GrLivArea',
        ymin: int = 0,
        ymax: int = 800000,
        figsize: tuple = (24, 12)
):
    fig, ax = plt.subplots(figsize=figsize)
    data = pd.concat([df[tar], df[var]], axis=1)
    if ymin is None or ymax is None:
        data.plot.scatter(x=var, y=tar,  ax=ax, grid=True)
    else:
        data.plot.scatter(x=var, y=tar, ylim=(ymin, ymax), ax=ax, grid=True)
    plt.show()


# boxplotでtargetとvarianceの関係を確認(varianceの数が少ない場合に見る)
def box_plot(
        df: pd.DataFrame,
        tar: str = 'SalePrice',
        var: str = 'OverallQual',
        ymin: int = 0,
        ymax: int = 800000,
        figsize: tuple = (24, 12)
):
    data = pd.concat([df[tar], df[var]], axis=1)
    fig = plt.figure(figsize=figsize)
    fig = sns.boxplot(x=var, y=tar, data=data)
    fig.axis(ymin=ymin, ymax=ymax)
    plt.xticks(rotation=90)
    plt.show()


# 相関行列でチェック
def corr_mat(df: pd.DataFrame, vmax: float = .8, figsize: tuple = (12, 12)):
    corrmat = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corrmat, vmax=vmax, square=True)
    plt.show()


# 相関がありそうな10個の変数のみ確認
def zoom_corr_mat(
        df: pd.DataFrame,
        tar: str = 'SalePrice',
        feat_num: int = 10,
        figsize: tuple = (10, 10)
):
    fig, ax = plt.subplots(figsize=figsize)
    corrmat = df.corr()
    cols = corrmat.nlargest(feat_num, tar)[tar].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(
        cm,
        cbar=True,
        annot=True,
        square=True,
        fmt='.2f',
        annot_kws={'size': feat_num},
        yticklabels=cols.values,
        xticklabels=cols.values
    )
    plt.show()


def pair_plot(
    df: pd.DataFrame,
    cols: List[str] = ['SalePrice', 'OverallQual', 'GrLivArea',
                       'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'],
    height: float = 1.5,
    save_fig_d: Path = None
):
    sns.set()
    sns.pairplot(df[cols], height=height)
    if save_fig_d is not None:
        plt.savefig(save_fig_d)
    plt.show()


def hist_norm_plot(
        df: pd.DataFrame,
        tar='SalePrice',
        save_fig_d: Path = None
):
    mu, std = norm.fit(df[tar])
    sns.histplot(df[tar], stat='density', kde=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    prob = norm.pdf(x, mu, std)
    plt.plot(x, prob, linewidth=2, color='black')
    if save_fig_d is not None:
        plt.savefig(save_fig_d)
    plt.show()


def prob_plot(
        df: pd.DataFrame,
        tar: str = 'SalePrice',
        figsize: tuple = (16, 12),
        save_fig_d: Path = None
):
    stats.probplot(df[tar], plot=plt)
    plt.grid()
    if save_fig_d is not None:
        plt.savefig(save_fig_d)
    plt.show()


def log_prob_plot(
    df: pd.DataFrame,
    tar: str = 'SalePrice',
    figsize: tuple = (16, 12),
    save_fig_d: Path = None
):
    stats.probplot(np.log(df[tar]), plot=plt)
    plt.grid()
    if save_fig_d is not None:
        plt.savefig(save_fig_d)
    plt.show()
